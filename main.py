import argparse
import torch
import os
import wandb
import torch.nn as nn
from config import Config
from train import (
    ICLEVRDataset, load_json, create_model, evaluate_model_with_labels, 
    get_inference_scheduler, generate_and_save_test_images, train_loop
)
from evaluator import evaluation_model

def parse_args():
    parser = argparse.ArgumentParser(description="Conditional DDPM for i-CLEVR dataset")
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode to run in')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint to load')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')

    return parser.parse_args()

def main():
    # 解析命令列參數
    args = parse_args()
    
    # 初始化配置
    if args.config:
        config = Config(args.config)
    else:
        config = Config()
    
    # 從命令列參數更新配置
    config.update_from_args(args)
    
    # 創建運行資料夾
    if args.mode == 'train':
        run_dir = config.create_run_folders()
        print(f"Created run directory: {run_dir}")
    
    # 初始化設備
    device = torch.device(config.device)
    
    # 載入物件映射
    objects_map = load_json(config.objects_json_path)
    
    # 初始化模型
    model, condition_projector, num_classes = create_model(config.to_dict(), device)
    
    # 初始化評估器
    evaluator = evaluation_model()
    evaluator.resnet18.to(device)
    evaluator.resnet18.eval()
    
    # 載入檢查點（若有指定）
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        condition_projector.load_state_dict(checkpoint['condition_projector_state_dict'])
        
        if 'epoch' in checkpoint and args.mode == 'train':
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    
    # 準備測試數據
    test_dataset = ICLEVRDataset(
        config.test_json_path, objects_map, 
        "", config.image_size, mode="test"
    )
    test_labels_one_hot_list = [test_dataset[i] for i in range(len(test_dataset))]
    test_labels_one_hot_tensor = torch.stack(test_labels_one_hot_list)
    
    new_test_dataset = ICLEVRDataset(
        config.new_test_json_path, objects_map, 
        "", config.image_size, mode="test"
    )
    new_test_labels_one_hot_list = [new_test_dataset[i] for i in range(len(new_test_dataset))]
    new_test_labels_one_hot_tensor = torch.stack(new_test_labels_one_hot_list)
    
    # 根據模式執行訓練或測試
    if args.mode == 'train':
        # 初始化 wandb
        if not args.no_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.to_dict(),
                save_code=True
            )
        
        # 創建訓練數據加載器
        train_dataset = ICLEVRDataset(
            config.train_json_path, objects_map, 
            config.images_base_path, config.image_size, 
            mode="train"
        )
        
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.train_batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True,
        )
        
        # 初始化優化器和學習率調度器
        from diffusers import DDPMScheduler
        from diffusers.optimization import get_cosine_schedule_with_warmup
        
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule
        )
        
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_projector.parameters()),
            lr=config.lr
        )
        
        # 如果恢復訓練，載入優化器狀態
        if args.checkpoint and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
        
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=(len(train_dataset) // config.train_batch_size * config.num_epochs),
        )
        
        # 如果恢復訓練，載入學習率調度器狀態
        if args.checkpoint and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print("Loaded learning rate scheduler state from checkpoint")
        
        # 執行訓練循環
        train_loop(
            config.to_dict(), model, condition_projector, noise_scheduler, 
            optimizer, lr_scheduler, train_dataloader,
            test_labels_one_hot_tensor, evaluator,
            start_epoch=start_epoch
        )
        
        if not args.no_wandb and wandb.run is not None:
            wandb.finish()
            
    else:  # 測試模式
        # 初始化推理調度器
        inference_scheduler = get_inference_scheduler(config.to_dict())
        
        # 執行模型評估
        accuracy_test, accuracy_new_test = evaluate_model_with_labels(
            model=model, 
            condition_projector=condition_projector, 
            noise_scheduler=inference_scheduler,
            evaluator_obj=evaluator,
            test_labels_one_hot_tensor=test_labels_one_hot_tensor,
            new_test_labels_one_hot_tensor=new_test_labels_one_hot_tensor,
            epoch_num=0,
            config=config.to_dict()
        )
        
        print(f"\nFinal Results:")
        print(f"Accuracy on test.json: {accuracy_test:.4f}")
        print(f"Accuracy on new_test.json: {accuracy_new_test:.4f}")
        
        # 生成並保存測試圖像
        generate_and_save_test_images(
            model=model,
            condition_projector=condition_projector,
            scheduler=inference_scheduler,
            evaluator_obj=evaluator,
            test_dataset=test_labels_one_hot_list,
            new_test_dataset=new_test_labels_one_hot_list,
            config=config.to_dict()
        )
        
        print("Testing completed successfully")

if __name__ == "__main__":
    main()