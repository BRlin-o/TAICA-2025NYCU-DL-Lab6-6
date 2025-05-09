import os
import json
from typing import Optional, Union
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from diffusers import UNet2DConditionModel, DDPMScheduler, DPMSolverMultistepScheduler

def build_model_path(base_path: str, suffix: str) -> str:
    """
    Utility to insert a suffix **before** the file extension.

    Example:
        build_model_path("model.pth", "_best") -> "model_best.pth"
    """
    root, ext = os.path.splitext(base_path)
    return f"{root}{suffix}{ext}"

class Config(dict):
    """
    統一設定物件：
    - 預設參數集中於此
    - 同時支援 config["key"] 及 config.key
    - 初始化時自動建立 results_folder / images/test / images/new_test
    """
    def __init__(self, **overrides):
        defaults = {
            # ── experiment & logging ──
            "wandb_project": "conditional-ddpm-lab6-6",
            "wandb_run_name": "ddpm-dpm-solver-run",
            "log_images_freq": 10,
            # ── data & training ──
            "image_size": 64,
            "train_batch_size": 64,
            "eval_batch_size": 64,
            "num_epochs": 200,
            "lr": 1e-4,
            "num_train_timesteps": 1000,
            "num_inference_steps": 50,
            # ── sampling ──
            "inference_scheduler": "ddpm",   # or "dpm_solver++"
            "dpm_solver_steps": 20,
            # ── bookkeeping ──
            "gradient_accumulation_steps": 1,
            "log_interval": 50,
            "save_image_epochs": 1, # 15
            "save_model_epochs": 1, # 25
            "eval_epochs": 1, # 5
            # ── paths ──
            "results_folder": "./results_lab6",
            "model_save_path": "./ddpm_conditional_iclevr.pth",
            "images_base_path": "./iclevr/",
            "train_json_path": "./train.json",
            "test_json_path": "./test.json",
            "new_test_json_path": "./new_test.json",
            "objects_json_path": "./objects.json",
            "evaluator_ckpt_path": "./checkpoint.pth",
            # ── model & guidance ──
            "condition_embed_dim": 128,
            "use_classifier_guidance": True,
            "guidance_scale": 2.0,
            # ── hardware ──
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        defaults.update(overrides)          # 覆寫
        super().__init__(defaults)

        # 自動建立資料夾
        os.makedirs(self["results_folder"], exist_ok=True)
        os.makedirs("images/test", exist_ok=True)
        os.makedirs("images/new_test", exist_ok=True)

    # 允許 config.key 形式
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def to_dict(self):
        """便於 wandb.log(config.to_dict())"""
        return dict(self)

class ICLEVRDataset(Dataset):
    def __init__(self, json_path, objects_map, images_base_path, image_size, mode="train"):
        self.data = load_json(json_path)
        self.objects_map = objects_map
        self.num_classes = len(objects_map)
        self.images_base_path = images_base_path
        self.mode = mode # "train", "test"
        self.tag = mode  # keep a simple tag for later use

        if self.mode == "train":
            self.image_files = list(self.data.keys())
            self.labels = list(self.data.values())
        else: # "test" or "new_test"
            self.labels = self.data
            self.image_files = None

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # Scales to [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Scales to [-1, 1]
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

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def denormalize(tensor_images): # Convert from [-1, 1] to [0, 1]
    return (tensor_images / 2.0) + 0.5

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