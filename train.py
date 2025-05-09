import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from PIL import Image
import json
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

CONFIG = {
    "wandb_project": "conditional-ddpm-lab6-6",  # wandb 專案名稱
    "wandb_run_name": "ddpm-dpm-solver-run",  # 運行名稱
    "log_images_freq": 10,  # 每隔多少個 epoch 記錄圖像
    "image_size": 64,
    "train_batch_size": 64,
    "eval_batch_size": 64,
    "num_epochs": 200,
    "lr": 1e-4,
    "num_train_timesteps": 1000,
    "num_inference_steps": 50,
    "inference_scheduler": "dpm_solver++",  # 可選: "ddpm", "dpm_solver++"
    "dpm_solver_steps": 20,  # DPM-Solver++ 只需要較少的步數
    "gradient_accumulation_steps": 1,
    "log_interval": 50,
    "save_image_epochs": 20,
    "save_model_epochs": 50,
    "eval_epochs": 10, 
    "results_folder": "./results_lab6",
    "model_save_path": "./ddpm_conditional_iclevr.pth",
    "condition_embed_dim": 128,
    "use_classifier_guidance": True,
    "guidance_scale": 2.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_json_path": "./train.json",
    "test_json_path": "./test.json",
    "new_test_json_path": "./new_test.json",
    "objects_json_path": "./objects.json",
    "images_base_path": "./iclevr/",
    "evaluator_ckpt_path": "./checkpoint.pth"
}

os.makedirs(CONFIG["results_folder"], exist_ok=True)
device = torch.device(CONFIG["device"])

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def denormalize(tensor_images): # Convert from [-1, 1] to [0, 1]
    return (tensor_images / 2.0) + 0.5

objects_map = load_json(CONFIG["objects_json_path"])
idx_to_object = {v: k for k, v in objects_map.items()}
num_classes = len(objects_map)

def get_inference_scheduler(config):
    """返回用於推理的採樣排程器"""
    if config["inference_scheduler"] == "dpm_solver++":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=config["num_train_timesteps"],
            beta_schedule="squaredcos_cap_v2",
            algorithm_type="dpmsolver++",  # 使用 dpmsolver++ 算法
            solver_order=2,  # 可以是 1, 2 或 3，越高精度越好但計算量更大
        )
    else:  # 默認使用 DDPM
        return DDPMScheduler(
            num_train_timesteps=config["num_train_timesteps"],
            beta_schedule="squaredcos_cap_v2"
        )

class ICLEVRDataset(Dataset):
    def __init__(self, json_path, objects_map, images_base_path, image_size, mode="train"):
        self.data = load_json(json_path)
        self.objects_map = objects_map
        self.num_classes = len(objects_map)
        self.images_base_path = images_base_path
        self.mode = mode # "train", "test"

        if self.mode == "train":
            self.image_files = list(self.data.keys())
            self.labels = list(self.data.values())
        else: # "test" or "new_test"
            self.labels = self.data
            self.image_files = None

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # Scales to [0, 1]
            transforms.Normalize([0.5], [0.5]) # Scales to [-1, 1]
        ])

    def __len__(self):
        if self.mode == "train":
            return len(self.image_files)
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        labels_text = self.labels[idx]
        label_one_hot = torch.zeros(self.num_classes, dtype=torch.float)
        for obj_name in labels_text:
            label_one_hot[self.objects_map[obj_name]] = 1.0

        if self.mode == "train":
            img_name = self.image_files[idx]
            img_path = os.path.join(self.images_base_path, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = self.transform(image)
            except FileNotFoundError:
                print(f"Warning: Image file not found {img_path}. Returning zeros.")
                image_tensor = torch.zeros((3, CONFIG["image_size"], CONFIG["image_size"]))
            return image_tensor, label_one_hot
        else: # For test mode, only return the labels. Images will be generated.
            return label_one_hot

model = UNet2DConditionModel(
    sample_size=CONFIG["image_size"],
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512), # Standard U-Net architecture
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "CrossAttnDownBlock2D", # Add CrossAttention for conditioning
        "CrossAttnDownBlock2D",
    ),
    up_block_types=(
        "CrossAttnUpBlock2D", # Add CrossAttention for conditioning
        "CrossAttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    cross_attention_dim=CONFIG["condition_embed_dim"], # Dimension of condition embedding
).to(device)

# Condition Projection Layer
condition_projector = nn.Linear(num_classes, CONFIG["condition_embed_dim"]).to(device)

# Noise Scheduler (from diffusers)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=CONFIG["num_train_timesteps"],
    beta_schedule='squaredcos_cap_v2' # Often gives good results
    # beta_schedule='linear' # Simpler alternative
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

original_evaluator_ckpt_path = './checkpoint.pth' # Path used inside evaluator.py
evaluator = evaluation_model()
evaluator.resnet18.to(device) # Move evaluator's model to device for guidance
evaluator.resnet18.eval() # Set to eval mode

@torch.no_grad()
def evaluate_model(
    dataset_name: str,
    ddpm_model: UNet2DConditionModel, cond_projector: nn.Linear, 
    scheduler: DDPMScheduler, evaluator_obj: evaluation_model,
    test_labels_one_hot_tensor, epoch_num, config
):
    ddpm_model.eval()
    cond_projector.eval()

    # 獲取推理排程器
    inference_scheduler = get_inference_scheduler(config)
    
    # 根據採樣器類型設置適當的步數
    if isinstance(inference_scheduler, DPMSolverMultistepScheduler):
        inference_steps = config["dpm_solver_steps"]  # DPM-Solver++ 使用較少步數
    else:
        inference_steps = config["num_inference_steps"]  # DDPM 使用標準步數

    all_generated_images = []
    # Use a DataLoader for the labels to handle batching
    eval_labels_dataloader = DataLoader(test_labels_one_hot_tensor, batch_size=config["eval_batch_size"], shuffle=False)

    for batch_labels in tqdm(eval_labels_dataloader, desc=f"Generating for eval epoch {epoch_num}"):
        batch_labels = batch_labels.to(config["device"]) # Ensure labels are on the correct device
        generated_batch = generate_images(
            ddpm_model, cond_projector, inference_scheduler, evaluator_obj,
            batch_labels, num_images_per_prompt=1,
            use_guidance=config["use_classifier_guidance"], guidance_scale=config["guidance_scale"],
            device=config["device"], num_inference_steps=inference_steps
        )
        all_generated_images.append(generated_batch.cpu())
    
    all_generated_images_tensor = torch.cat(all_generated_images, dim=0)

    # Ensure ground truth labels are also on CPU for the evaluator's compute_acc
    accuracy = evaluator_obj.eval(all_generated_images_tensor.to(config["device"]), test_labels_one_hot_tensor.cpu())

    save_path = os.path.join(config["results_folder"], f"epoch_{epoch_num}_eval_{dataset_name}_samples.png")
    grid_eval = make_grid(denormalize(all_generated_images_tensor), nrow=8)
    save_image(grid_eval, save_path)
    print(f"Saved epoch {epoch_num} evaluation grid to {save_path}")

    wandb.log({
        f"eval/{dataset_name}_samples": wandb.Image(grid_eval, caption=f"{dataset_name} - Epoch {epoch_num}"),
        f"eval/{dataset_name}_accuracy": accuracy,
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
                if is_dpm_solver:
                    # DPM-Solver++ 對應的預測原始圖像方法
                    # 需要按照 DPM-Solver++ 的方式計算，這是一個近似
                    alpha_t = scheduler.alphas_cumprod[t]
                    x0_pred = (model_input - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                else:
                    # DDPM 的預測方式
                    alpha_prod_t = scheduler.alphas_cumprod[t]
                    x0_pred = (model_input - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()


                x0_pred_grad = x0_pred.detach().requires_grad_(True)

                # 2. Get guidance gradient from classifier
                # Evaluator expects normalized images in [-1, 1], which x0_pred should be.
                logits = evaluator_obj.resnet18(x0_pred_grad) # Use the resnet18 directly
                
                # Calculate log probability of target labels
                # For multi-label, BCE loss is common. We want to maximize log P(y|x).
                # So, we can use sum of log-sigmoid for present classes.
                log_probs = F.logsigmoid(logits) # (batch_size, num_classes)
                
                # We want to increase the probability of the true labels
                # eff_target_labels_one_hot is (batch_size, num_classes)
                selected_log_probs = (eff_target_labels_one_hot * log_probs).sum() # Sum over classes and batch

                grad = torch.autograd.grad(selected_log_probs, x0_pred_grad)[0]

            # 3. Adjust noise prediction
            # grad is d(logP)/dx0. We need to map this to d(logP)/d_epsilon
            if is_dpm_solver:
                # DPM-Solver++ 的引導方式稍有不同
                alpha_t = scheduler.alphas_cumprod[t]
                noise_pred = noise_pred - (1 - alpha_t).sqrt() * guidance_scale * grad
            else:
                # DDPM 的引導方式
                alpha_prod_t = scheduler.alphas_cumprod[t]
                noise_pred = noise_pred - (1 - alpha_prod_t).sqrt() * guidance_scale * grad

        # Compute previous image: x_t -> x_t-1
        images = scheduler.step(noise_pred, t, images).prev_sample
        
        if return_denoising_process and (t % (num_inference_steps // 8) == 0 or t == scheduler.timesteps[-1]):
             denoising_process_images.append(images[0].unsqueeze(0).clone().cpu())


    if return_denoising_process:
        # also add final clear image
        if not (scheduler.timesteps[-1] % (num_inference_steps // 8) == 0 or scheduler.timesteps[-1] == scheduler.timesteps[-1]):
             denoising_process_images.append(images[0].unsqueeze(0).clone().cpu())
        return images, torch.cat(denoising_process_images, dim=0)
    
    return images

# Evaluate with test labels and new test labels

def evaluate_model_with_labels(
    model: UNet2DConditionModel, condition_projector: nn.Linear, noise_scheduler, 
    evaluator_obj, test_labels_one_hot_tensor, 
    new_test_labels_one_hot_tensor, epoch_num, config
):
    print(f"\n--- Evaluating model at epoch {epoch_num} ---")
    # Evaluate on test.json
    accuracy_test = evaluate_model(
        dataset_name="test.json",
        ddpm_model=model, 
        cond_projector=condition_projector, 
        scheduler=noise_scheduler, 
        evaluator_obj=evaluator_obj,
        test_labels_one_hot_tensor=test_labels_one_hot_tensor, 
        epoch_num=epoch_num, 
        config=config
    )
    print(f"Epoch {epoch_num} - Accuracy on test.json: {accuracy_test:.4f}")

    # Evaluate on new_test.json
    accuracy_new_test = evaluate_model(
        dataset_name="new_test.json",
        ddpm_model=model, 
        cond_projector=condition_projector, 
        scheduler=noise_scheduler, 
        evaluator_obj=evaluator_obj,
        test_labels_one_hot_tensor=new_test_labels_one_hot_tensor, 
        epoch_num=epoch_num, 
        config=config
    )
    print(f"Epoch {epoch_num} - Accuracy on new_test.json: {accuracy_new_test:.4f}")
    return accuracy_test, accuracy_new_test

# --- Training Function ---
def train_loop(config, model, condition_projector, noise_scheduler, optimizer, lr_scheduler, train_dataloader,
               test_labels_eval_tensor, evaluator_obj): # <<< Added test_labels_eval_tensor and evaluator_obj
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
            loss = F.mse_loss(noise_pred, noise)
            
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

            # 記錄圖像到 wandb
            wandb.log({
                "generated_samples": wandb.Image(grid, caption=f"Epoch {epoch+1} Samples"),
                "epoch": epoch
            })

            model.train() # Set back to train mode
            condition_projector.train()


        # <<< Periodic Evaluation >>>
        if (epoch + 1) % config["eval_epochs"] == 0 or epoch == config["num_epochs"] - 1:
            current_eval_accuracy = evaluate_model(
                dataset_name="test.json",
                ddpm_model=model, 
                cond_projector=condition_projector, 
                scheduler=inference_scheduler , 
                evaluator_obj=evaluator_obj,
                test_labels_one_hot_tensor=test_labels_eval_tensor, 
                epoch_num=epoch + 1, 
                config=config
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
                model_save_path = config["model_save_path"] + "_best_eval"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'condition_projector_state_dict': condition_projector.state_dict(),
                    'epoch': epoch,
                    'accuracy': best_eval_accuracy
                }, model_save_path)
            model.train() # Ensure model is back in train mode after evaluation
            condition_projector.train()


        if (epoch + 1) % config["save_model_epochs"] == 0 or epoch == config["num_epochs"] - 1:
            model_path = config["model_save_path"] + f"_epoch{epoch+1}"
            torch.save({
                'model_state_dict': model.state_dict(),
                'condition_projector_state_dict': condition_projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step
            }, model_path)
            print(f"Saved model checkpoint at epoch {epoch+1}")
    
    progress_bar.close()
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


# --- Main Execution ---
if __name__ == "__main__":
    run = wandb.init(
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run_name"],
        config=CONFIG,  # 自動記錄所有配置
        save_code=True  # 保存代碼版本
    )
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
    test_labels_one_hot_list_for_eval = [test_dataset_for_eval[i] for i in range(len(test_dataset_for_eval))]
    # test_labels_eval_tensor = torch.stack(test_labels_one_hot_list_for_eval).to(device) # Move to device inside evaluate_model or pass device
    test_labels_eval_tensor = torch.stack(test_labels_one_hot_list_for_eval) # Keep on CPU, move to device in batches in evaluate_model


    # --- 2. Train Model (or load if exists) ---
    TRAIN_MODEL = True # Set to False to skip training and load
    model_load_path = CONFIG["model_save_path"] # Default final model
    # model_load_path = CONFIG["model_save_path"] + "_best_eval" # Optionally load best eval model
    
    if TRAIN_MODEL:
        print("Starting training...")
        # Pass test_labels_eval_tensor and evaluator to train_loop
        train_loop(CONFIG, model, condition_projector, noise_scheduler, optimizer, lr_scheduler, train_dataloader,
                   test_labels_eval_tensor, evaluator) # <<< Pass evaluator
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
    test_labels_one_hot_list = [test_dataset[i] for i in range(len(test_dataset))]
    test_labels_one_hot_tensor = torch.stack(test_labels_one_hot_list) # CPU tensor

    final_accuracy_test = evaluate_model(
        dataset_name="test.json",
        ddpm_model=model, 
        cond_projector=condition_projector, 
        scheduler=inference_scheduler, 
        evaluator_obj=evaluator, 
        test_labels_one_hot_tensor=test_labels_one_hot_tensor, 
        epoch_num=CONFIG["num_epochs"], 
        config=CONFIG
    ) # Pass full CONFIG
    print(f"Final Accuracy on test.json: {final_accuracy_test:.4f}")
    plt.figure(figsize=(3,1))
    plt.text(0.5, 0.5, f"Final test.json Accuracy: {final_accuracy_test:.4f}", ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.savefig(os.path.join(CONFIG["results_folder"], "accuracy_final_test_json.png"))
    # plt.show() # Can be blocking

    # New_test.json evaluation
    new_test_dataset = ICLEVRDataset(CONFIG["new_test_json_path"], objects_map, "", CONFIG["image_size"], mode="test")
    new_test_labels_one_hot_list = [new_test_dataset[i] for i in range(len(new_test_dataset))]
    new_test_labels_one_hot_tensor = torch.stack(new_test_labels_one_hot_list) # CPU tensor

    final_accuracy_new_test = evaluate_model( ## 
        dataset_name="new_test.json",
        ddpm_model=model, 
        cond_projector=condition_projector, 
        scheduler=inference_scheduler, 
        evaluator_obj=evaluator,
        test_labels_one_hot_tensor=new_test_labels_one_hot_tensor, 
        epoch_num=CONFIG["num_epochs"], 
        config=CONFIG
    ) # Pass full CONFIG
    print(f"Final Accuracy on new_test.json: {final_accuracy_new_test:.4f}")
    plt.figure(figsize=(3,1))
    plt.text(0.5, 0.5, f"Final new_test.json Accuracy: {final_accuracy_new_test:.4f}", ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.savefig(os.path.join(CONFIG["results_folder"], "accuracy_final_new_test_json.png"))
    # plt.show()

    wandb.run.summary.update({
        "final_test_accuracy": final_accuracy_test,
        "final_new_test_accuracy": final_accuracy_new_test,
        "dpm_solver_steps": CONFIG["dpm_solver_steps"] if CONFIG["inference_scheduler"] == "dpm_solver++" else None,
        "num_inference_steps": CONFIG["num_inference_steps"],
        "guidance_scale": CONFIG["guidance_scale"]
    })

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
    save_image(grid_denoising, os.path.join(CONFIG["results_folder"], "denoising_process.png"))
    print(f"Saved denoising process grid to {CONFIG['results_folder']}/denoising_process.png")

    print("\nLab 6 execution finished.")
    wandb.finish()
