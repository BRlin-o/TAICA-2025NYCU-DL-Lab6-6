import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from PIL import Image
import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb  # 添加 wandb 導入
# Diffusers specific imports
from diffusers import UNet2DConditionModel, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# Import the evaluator
from evaluator import evaluation_model # Make sure evaluator.py is in the same directory or accessible
from utils import build_model_path, load_json, denormalize, get_inference_scheduler
from utils import Config, ICLEVRDataset

CONFIG = Config()

device = torch.device(CONFIG["device"])

objects_map = load_json(CONFIG["objects_json_path"])
idx_to_object = {v: k for k, v in objects_map.items()}
num_classes = len(objects_map)

# --- Unified U-Net with Condition Projector ---
class UNetConditionModel(nn.Module):
    """
    Combines the conditional U‑Net and its label‑projection layer into a reusable
    module.  The rest of the pipeline can interact with just this class.
    """
    def __init__(self, num_classes: int, config):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=config["image_size"],
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
            cross_attention_dim=config["condition_embed_dim"],
        )
        # Label‑projection layer
        self.projector = nn.Linear(num_classes, config["condition_embed_dim"])

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels_one_hot: torch.Tensor):
        cond_embeds = self.projector(labels_one_hot).unsqueeze(1)
        return self.unet(x, t, encoder_hidden_states=cond_embeds).sample

### Unified model wrapper
# Instantiate the unified model wrapper
model_wrapper = UNetConditionModel(num_classes, CONFIG).to(device)
# Keep the original variable names for minimal downstream change
model = model_wrapper.unet
condition_projector = model_wrapper.projector

# Noise Scheduler (from diffusers)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=CONFIG["num_train_timesteps"],
    beta_schedule="squaredcos_cap_v2",
    prediction_type="v_prediction",        # use v‑prediction (Karras 2022)
)

# Optimizer
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(condition_projector.parameters()),
    lr=CONFIG["lr"]
)

# LR Scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(ICLEVRDataset(CONFIG["train_json_path"], objects_map, CONFIG["images_base_path"], CONFIG["image_size"])) // CONFIG["train_batch_size"] * CONFIG["num_epochs"]),
)

evaluator = evaluation_model()
evaluator.resnet18.to(device) # Move evaluator's model to device for guidance
evaluator.resnet18.eval()

@torch.no_grad()
def evaluate_model(
    dataloader: DataLoader,
    ddpm_model: UNet2DConditionModel,
    cond_projector: nn.Linear,
    scheduler: DDPMScheduler,
    evaluator_obj: evaluation_model,
    epoch_num: int,
    config,
    save_individual_images: bool = False,   # save PNGs for each sample
    use_wnadb=False,
):
    ddpm_model.eval()
    cond_projector.eval()
    tag = dataloader.dataset.mode  # "test" | "new_test"

    # 根據採樣器類型設置適當的步數
    if isinstance(scheduler, DPMSolverMultistepScheduler):
        inference_steps = config["dpm_solver_steps"]  # DPM-Solver++ 使用較少步數
    else:
        inference_steps = config["num_inference_steps"]  # DDPM 使用標準步數

    all_generated_images = []
    all_labels = []
    # ── optional per‑sample saving ──
    if save_individual_images:
        img_save_dir = os.path.join("images", tag)
        os.makedirs(img_save_dir, exist_ok=True)
        img_counter = 0
    for batch_labels in tqdm(dataloader, desc=f"Generating for eval epoch {epoch_num} ({tag})"):
        batch_labels = batch_labels.to(config["device"]) # Ensure labels are on the correct device
        generated_batch = generate_images(
            ddpm_model, cond_projector, scheduler, evaluator_obj,
            batch_labels, num_images_per_prompt=1,
            use_guidance=config["use_classifier_guidance"], guidance_scale=config["guidance_scale"],
            device=config["device"], num_inference_steps=inference_steps
        )
        if save_individual_images:
            for single_img in denormalize(generated_batch.cpu()):
                save_image(single_img, os.path.join(img_save_dir, f"{img_counter}.png"))
                img_counter += 1
        all_generated_images.append(generated_batch.cpu())
        all_labels.append(batch_labels.cpu())

    all_generated_images_tensor = torch.cat(all_generated_images, dim=0)

    accuracy = evaluator_obj.eval(
        all_generated_images_tensor.to(config["device"]),
        torch.cat(all_labels, dim=0).to(config["device"])
    )

    save_path = os.path.join(config["results_folder"], f"epoch_{epoch_num}_eval_{tag}_samples.png")
    grid_eval = make_grid(denormalize(all_generated_images_tensor), nrow=8)
    save_image(grid_eval, save_path)
    print(f"Saved epoch {epoch_num} evaluation grid to {save_path}")

    if use_wnadb:
        wandb.log({
            f"eval/{tag}_accuracy": accuracy,
            "epoch": epoch_num
        })

    return accuracy

# --- Sampling/Generation Function ---
@torch.no_grad()
def generate_images(ddpm_model, cond_projector, scheduler, evaluator_obj,
                    target_labels_one_hot, num_images_per_prompt,
                    use_guidance, guidance_scale, device, num_inference_steps,
                    return_denoising_process=False):
    ddpm_model.eval()
    cond_projector.eval()

    batch_size = target_labels_one_hot.shape[0] * num_images_per_prompt
    
    # Repeat labels for num_images_per_prompt
    eff_target_labels_one_hot = target_labels_one_hot.repeat_interleave(num_images_per_prompt, dim=0).to(device)
    condition_embeds = cond_projector(eff_target_labels_one_hot).unsqueeze(1) # (B*N, 1, embed_dim)

    # Initialize with random noise
    images = torch.randn(
        (batch_size, ddpm_model.config.in_channels, ddpm_model.config.sample_size, ddpm_model.config.sample_size)
    ).to(device)

    scheduler.set_timesteps(num_inference_steps)
    
    denoising_process_images = []
    if return_denoising_process:
        denoising_process_images.append(images[0].unsqueeze(0).clone().cpu()) # Store initial noise for one image

    # 判斷是否為 DPMSolverMultistepScheduler
    is_dpm_solver = isinstance(scheduler, DPMSolverMultistepScheduler)

    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        model_input = images # current noisy image

        # Predict noise model output
        noise_pred = ddpm_model(model_input, t, encoder_hidden_states=condition_embeds).sample

        if use_guidance:
            with torch.enable_grad():
                alpha_cumprod_t = scheduler.alphas_cumprod[t]
                sigma_t = (1 - alpha_cumprod_t).sqrt()

                if scheduler.config.prediction_type == "v_prediction":
                    # x₀ = α · x_t − σ · v̂
                    alpha_t = alpha_cumprod_t.sqrt()
                    x0_pred = alpha_t * model_input - sigma_t * noise_pred
                else:
                    # x₀ = (x_t − σ · ε̂) / α
                    x0_pred = (model_input - sigma_t * noise_pred) / alpha_cumprod_t.sqrt()

                x0_pred_grad = x0_pred.detach().requires_grad_(True)
                logits = evaluator_obj.resnet18(x0_pred_grad)
                log_probs = F.logsigmoid(logits)
                selected_log_probs = (eff_target_labels_one_hot * log_probs).sum()
                grad = torch.autograd.grad(selected_log_probs, x0_pred_grad)[0]

            # Map ∂L/∂x₀ to ∂L/∂prediction  (both ε‑ and v‑pred: factor is σ_t)
            noise_pred = noise_pred - sigma_t * guidance_scale * grad

        # Compute previous image: x_t -> x_t-1
        images = scheduler.step(noise_pred, t, images).prev_sample
        
        if return_denoising_process and (t % (num_inference_steps // 8) == 0 or t == scheduler.timesteps[-1]):
             denoising_process_images.append(images[0].unsqueeze(0).clone().cpu())


    if return_denoising_process:
        # also add final clear image
        denoising_process_images.append(images[0].unsqueeze(0).clone().cpu())
        return images, torch.cat(denoising_process_images, dim=0)
    
    return images

# ---------- Loss-weight utilities ----------
def compute_p2_loss_weight(scheduler, timesteps, gamma: float = 5.0, device=None):
    """
    手動計算 P2 / SNR-based loss weighting。
    公式同 diffusers >=0.34 的 `get_loss_weight()`：
        w_t = (SNR_t + 1) / (SNR_t + gamma)
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(device or timesteps.device)
    alphas = alphas_cumprod[timesteps]          # ᾱ_t
    snr = alphas / (1.0 - alphas)               # signal-to-noise ratio
    return (snr + 1.0) / (snr + gamma)

def train_loop(config, model, condition_projector, noise_scheduler, optimizer, lr_scheduler, train_dataloader,
               test_loader_eval, evaluator_obj):
    model.train()
    condition_projector.train()
    global_step = 0
    best_eval_accuracy = 0.0 # To save the best model based on eval

    # 獲取用於推理的排程器
    inference_scheduler = get_inference_scheduler(config)
    
    # 根據採樣器類型設置適當的步數
    if isinstance(inference_scheduler, DPMSolverMultistepScheduler):
        inference_steps = config["dpm_solver_steps"]  # DPM-Solver++ 使用較少步數
        print(f"使用 DPM-Solver++ 生成訓練中的示例圖像，步數: {inference_steps}")
    else:
        inference_steps = config["num_inference_steps"]  # DDPM 使用標準步數
        print(f"使用 DDPM 生成訓練中的示例圖像，步數: {inference_steps}")

    for epoch in range(config["num_epochs"]):
        model.train()
        condition_projector.train()
        epoch_loss = 0.0

        # 每個 epoch 重設進度條
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config['num_epochs']}")

        for step, batch in enumerate(train_dataloader):
            clean_images, labels_one_hot = batch
            clean_images = clean_images.to(device)
            labels_one_hot = labels_one_hot.to(device)

            noise = torch.randn(clean_images.shape).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            condition_embeds = condition_projector(labels_one_hot)
            condition_embeds_unet = condition_embeds.unsqueeze(1)
            noise_pred = model(noisy_images, timesteps, encoder_hidden_states=condition_embeds_unet).sample

            # --- P2 loss with v‑prediction ---
            target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
            loss = F.mse_loss(noise_pred, target, reduction="none")
            loss_weights = compute_p2_loss_weight(noise_scheduler, timesteps, gamma=5.0, device=device)
            loss = (loss * loss_weights.view(-1, 1, 1, 1)).mean()
            
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 記錄訓練指標到 wandb
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
                "global_step": global_step
            })

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            epoch_loss += loss.item()
            global_step +=1

        # 關閉當前 epoch 的進度條
        progress_bar.close()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        # 記錄每個 epoch 的平均損失
        wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch})

        if (epoch + 1) % config["save_image_epochs"] == 0 or epoch == config["num_epochs"] - 1:
            model.eval() # Set to eval mode for generation
            condition_projector.eval()
            print("Generating sample images for visual inspection...")

            fixed_labels_text = [["red sphere", "cyan cylinder", "cyan cube"], ["yellow cube"], ["blue sphere", "green cylinder"]]
            fixed_labels_one_hot = []
            for l_text in fixed_labels_text:
                lh = torch.zeros(num_classes)
                for obj_name in l_text: lh[objects_map[obj_name]] = 1.0
                fixed_labels_one_hot.append(lh)
            fixed_labels_one_hot = torch.stack(fixed_labels_one_hot).to(device)

            generated_images = generate_images(
                model, condition_projector, inference_scheduler, evaluator_obj, # Pass evaluator_obj
                fixed_labels_one_hot, num_images_per_prompt=1,
                use_guidance=config["use_classifier_guidance"], guidance_scale=config["guidance_scale"],
                device=device, num_inference_steps=config["num_inference_steps"]
            )
            generated_images_denorm = denormalize(generated_images.cpu())
            grid = make_grid(generated_images_denorm, nrow=len(fixed_labels_text))

            save_path = os.path.join(config["results_folder"], f"epoch_{epoch+1}_samples.png")
            save_image(grid, save_path)
            print(f"Saved sample images to {save_path}")

            model.train() # Set back to train mode
            condition_projector.train()


        # <<< Periodic Evaluation >>>
        if (epoch + 1) % config["eval_epochs"] == 0 or epoch == config["num_epochs"] - 1:
            current_eval_accuracy = evaluate_model(
                test_loader_eval,
                ddpm_model=model,
                cond_projector=condition_projector,
                scheduler=inference_scheduler,
                evaluator_obj=evaluator_obj,
                epoch_num=epoch + 1,
                config=config,
                use_wnadb=True,
            )
            # 記錄評估指標到 wandb
            wandb.log({
                "eval/eval_accuracy": current_eval_accuracy,
                "epoch": epoch
            })

            if current_eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = current_eval_accuracy
                wandb.log({"eval/best_eval_accuracy": best_eval_accuracy})
                print(f"New best evaluation accuracy: {best_eval_accuracy:.4f}. Saving model.")
                model_save_path = build_model_path(config["model_save_path"], "_best_eval")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'condition_projector_state_dict': condition_projector.state_dict(),
                    'epoch': epoch,
                    'accuracy': best_eval_accuracy
                }, model_save_path)
            model.train() # Ensure model is back in train mode after evaluation
            condition_projector.train()


        if (epoch + 1) % config["save_model_epochs"] == 0 or epoch == config["num_epochs"] - 1:
            model_path = build_model_path(config["model_save_path"], f"_epoch{epoch+1}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'condition_projector_state_dict': condition_projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step
            }, model_path)
            print(f"Saved model checkpoint at epoch {epoch+1}")
    
    final_model_path = config["model_save_path"]
    torch.save({
        'model_state_dict': model.state_dict(),
        'condition_projector_state_dict': condition_projector.state_dict(),
    }, final_model_path)
    print(f"Training finished. Saved final model to {config['model_save_path']}")

    wandb.run.summary.update({
        "best_eval_accuracy": best_eval_accuracy,
        "total_epochs": config["num_epochs"],
        "final_epoch_loss": avg_epoch_loss
    })

## 吃arg 但只吃train or evaluate
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train or evaluate the model.")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train",
                        help="Mode to run: 'train' to train the model, 'evaluate' to evaluate the model.")
    args = parser.parse_args()
    return args

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    # --- 1. Prepare Dataset ---
    print("Preparing dataset...")
    train_dataset = ICLEVRDataset(CONFIG["train_json_path"], objects_map, CONFIG["images_base_path"], CONFIG["image_size"], mode="train")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["train_batch_size"], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,# Added pin_memory
        persistent_workers=True,
    ) 
    
    # Test datasets (only labels)
    test_dataset_for_eval = ICLEVRDataset(CONFIG["test_json_path"], objects_map, "", CONFIG["image_size"], mode="test")
    test_loader_eval = DataLoader(
        test_dataset_for_eval,
        batch_size=CONFIG["eval_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    # --- 2. Train Model (or load if exists) ---
    model_load_path = CONFIG["best_model_save_path"] # Default final model
    
    if args.mode == "train":
        run = wandb.init(
            project=CONFIG["wandb_project"],
            name=CONFIG["wandb_run_name"],
            config=CONFIG,  # 自動記錄所有配置
            save_code=True  # 保存代碼版本
        )
        print("Starting training...")
        train_loop(CONFIG, model, condition_projector, noise_scheduler, optimizer, lr_scheduler, train_dataloader,
                   test_loader_eval, evaluator)
        wandb.finish()
    else:
        print(f"Loading pre-trained model from {model_load_path}") # Use model_load_path
        if os.path.exists(model_load_path):
            checkpoint = torch.load(model_load_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            condition_projector.load_state_dict(checkpoint['condition_projector_state_dict'])
            # Optionally load optimizer, lr_scheduler, epoch if continuing training
            # if 'optimizer_state_dict' in checkpoint:
            #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # if 'lr_scheduler_state_dict' in checkpoint:
            #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            # if 'epoch' in checkpoint:
            #     start_epoch = checkpoint['epoch'] + 1 # resume from next epoch
            #     print(f"Resuming training from epoch {start_epoch}")
            model.to(device)
            condition_projector.to(device)
            print("Model loaded successfully.")
        else:
            print(f"Pre-trained model not found at {model_load_path}. Please train first or check path.")
            exit()

    # --- 3. Final Evaluation ---
    print("\n--- Final Evaluation ---")
    # Load the best model for final evaluation if desired, or use the last one.
    # If you saved "_best_eval", you might want to load it here.
    # For simplicity, we'll use the model currently in memory (either last epoch or loaded).
    inference_scheduler = get_inference_scheduler(CONFIG)

    # 如果使用 DPM-Solver++，顯示相關信息
    if isinstance(inference_scheduler, DPMSolverMultistepScheduler):
        print(f"使用 DPM-Solver++ 進行採樣，步數: {CONFIG['dpm_solver_steps']}")
        inference_steps = CONFIG["dpm_solver_steps"]
    else:
        print(f"使用 DDPM 進行採樣，步數: {CONFIG['num_inference_steps']}")
        inference_steps = CONFIG["num_inference_steps"]
    
    # Test.json evaluation
    test_dataset = ICLEVRDataset(CONFIG["test_json_path"], objects_map, "", CONFIG["image_size"], mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["eval_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    final_accuracy_test = evaluate_model(
        test_loader,
        ddpm_model=model,
        cond_projector=condition_projector,
        scheduler=inference_scheduler,
        evaluator_obj=evaluator,
        epoch_num=CONFIG["num_epochs"],
        config=CONFIG,
        save_individual_images=True
    )
    print(f"Final Accuracy on test.json: {final_accuracy_test:.4f}")

    # New_test.json evaluation
    new_test_dataset = ICLEVRDataset(CONFIG["new_test_json_path"], objects_map, "", CONFIG["image_size"], mode="new_test")
    new_test_loader = DataLoader(
        new_test_dataset,
        batch_size=CONFIG["eval_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    final_accuracy_new_test = evaluate_model(
        new_test_loader,
        ddpm_model=model,
        cond_projector=condition_projector,
        scheduler=inference_scheduler,
        evaluator_obj=evaluator,
        epoch_num=CONFIG["num_epochs"],
        config=CONFIG,
        save_individual_images=True
    )
    print(f"Final Accuracy on new_test.json: {final_accuracy_new_test:.4f}")

    # --- 4. Show Denoising Process ---
    # (This part remains the same, ensure model is in eval mode if not already)
    model.eval()
    condition_projector.eval()
    print("\nGenerating denoising process visualization...")
    denoising_labels_text = ["red sphere", "cyan cylinder", "cyan cube"]
    denoising_label_one_hot = torch.zeros(num_classes)
    for obj_name in denoising_labels_text:
        denoising_label_one_hot[objects_map[obj_name]] = 1.0
    denoising_label_one_hot = denoising_label_one_hot.unsqueeze(0).to(device)

    _, denoising_steps_images = generate_images(
        model, condition_projector, inference_scheduler, evaluator,
        denoising_label_one_hot, num_images_per_prompt=1,
        use_guidance=CONFIG["use_classifier_guidance"], guidance_scale=CONFIG["guidance_scale"],
        device=device, num_inference_steps=CONFIG["num_inference_steps"],
        return_denoising_process=True
    )
    
    if denoising_steps_images.shape[0] < 8 :
        print(f"Warning: Denoising process captured only {denoising_steps_images.shape[0]} images. Will display all.")
    
    grid_denoising = make_grid(denormalize(denoising_steps_images), nrow=denoising_steps_images.shape[0])
    os.makedirs("images", exist_ok=True)
    save_image(grid_denoising, os.path.join("images", "denoising_process.png"))
    print(f"Saved denoising process grid to images/denoising_process.png")

    print("\nLab 6 execution finished.")
    
