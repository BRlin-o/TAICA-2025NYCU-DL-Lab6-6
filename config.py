import torch
import os
import json
import datetime

class Config:
    def __init__(self, config_path=None):
        # 基本路徑設定
        self.results_folder = "./results_lab6"
        self.model_save_path = "./ddpm_conditional_iclevr.pth"
        self.train_json_path = "./train.json"
        self.test_json_path = "./test.json"
        self.new_test_json_path = "./new_test.json"
        self.objects_json_path = "./objects.json"
        self.images_base_path = "./iclevr/"
        self.evaluator_ckpt_path = "./checkpoint.pth"
        
        # wandb 設定
        self.wandb_project = "conditional-ddpm-lab6-6"
        self.wandb_run_name = "ddpm-dpm-solver-run"
        self.log_images_freq = 10
        
        # 資料與批次設定
        self.image_size = 64
        self.train_batch_size = 64
        self.eval_batch_size = 64
        
        # 訓練參數
        self.num_epochs = 200
        self.lr = 1e-4
        self.gradient_accumulation_steps = 1
        self.log_interval = 50
        self.save_image_epochs = 1
        self.save_model_epochs = 1
        self.eval_epochs = 1
        self.warmup_steps = 500
        
        # 模型參數
        self.condition_embed_dim = 128
        
        # 擴散參數
        self.num_train_timesteps = 1000
        self.num_inference_steps = 50
        self.inference_scheduler = "ddpm"  # 可選: "ddpm", "dpm_solver++"
        self.dpm_solver_steps = 20
        self.beta_schedule = "squaredcos_cap_v2"
        
        # 導向參數
        self.use_classifier_guidance = True
        self.guidance_scale = 2.0
        
        # 硬體參數
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 如果提供了配置路徑，則從檔案加載
        if config_path:
            self.load_config(config_path)
            
    def __getitem__(self, key):
        """允許像字典一樣使用方括號訪問屬性"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """允許像字典一樣使用方括號設置屬性"""
        setattr(self, key, value)
    
    def get(self, key, default=None):
        """與字典的get方法類似，如果屬性不存在則返回默認值"""
        return getattr(self, key, default)
    
    def to_dict(self):
        """將配置轉換為字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_config(self, path=None):
        """將配置保存為JSON檔案"""
        if path is None:
            os.makedirs(self.results_folder, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.results_folder, f"config_{timestamp}.json")
        
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"配置已保存到: {path}")
        return path
    
    def load_config(self, path):
        """從JSON檔案加載配置"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        print(f"已從 {path} 加載配置")
    
    def update_from_args(self, args):
        """從命令列參數更新配置"""
        args_dict = vars(args)
        for key, value in args_dict.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
    
    def create_run_folders(self):
        """創建必要的執行資料夾"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.results_folder, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
        
        # 更新結果資料夾
        self.results_folder = run_dir
        
        # 創建測試圖像資料夾
        os.makedirs(os.path.join("images", "test"), exist_ok=True)
        os.makedirs(os.path.join("images", "new_test"), exist_ok=True)
        
        # 保存配置到新的執行資料夾
        self.save_config(os.path.join(run_dir, "config.json"))
        
        return run_dir