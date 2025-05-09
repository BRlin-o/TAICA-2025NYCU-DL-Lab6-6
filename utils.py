import os
import torch        # ← 若沒有請加
from typing import Optional, Union

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
            "save_image_epochs": 15,
            "save_model_epochs": 25,
            "eval_epochs": 5,
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