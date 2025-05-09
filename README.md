# Lab6-6(Tim-BR改1)

## 環境設置與執行指令

### 1. 創建並激活虛擬環境（推薦）

```bash
# 創建虛擬環境
python -m venv .venv

# 激活環境 (Linux/Mac)
# source .venv/bin/activate

# 激活環境 (Windows)
.venv\Scripts\activate
```

### 2. 安裝依賴

```bash
# 安裝依賴
pip install -r requirements.txt
```

### 3. 準備數據

確保您的資料夾結構如下：

```
project_folder/
├── train.py               # 您的主程式碼
├── evaluator.py           # 提供的評估器
├── checkpoint.pth         # 評估器權重
├── train.json             # 訓練資料標籤
├── test.json              # 測試資料標籤
├── new_test.json          # 新測試資料標籤
├── objects.json           # 物件字典檔
└── iclevr/                # 圖像資料夾
    ├── CLEVR_train_001032_0.png
    ├── CLEVR_train_001032_1.png
    └── ...
```

### 4. 執行訓練

基本訓練指令：

```bash
python train.py
```

### 5. 執行評估或僅生成模式

如果您想跳過訓練直接加載預訓練模型進行評估，可以修改 `train.py` 中的 `TRAIN_MODEL` 變數：

```python
# 在 train.py 中找到這一行並修改
TRAIN_MODEL = False  # 設為 False 跳過訓練
```

然後執行：

```bash
python train.py
```

### 6. 自定義配置（可選）

如果您想使用不同的配置參數運行，可以修改 `CONFIG` 字典中的參數，例如：

```python
# 修改 CONFIG 中的關鍵參數
CONFIG = {
    # ...現有配置...
    "num_epochs": 100,         # 減少訓練週期
    "inference_scheduler": "dpm_solver++",  # 使用 DPM-Solver++
    "dpm_solver_steps": 20,     # 設定 DPM-Solver++ 步數
    "guidance_scale": 3.0,      # 提高引導強度
    # ...其他配置...
}
```

### 7. 調整GPU使用（可選）

如果您有多個GPU或想要限制GPU使用，可以在運行前設置環境變數：

```bash
# 使用指定GPU (例如第0號GPU)
CUDA_VISIBLE_DEVICES=0 python train.py

# 使用多個GPU (例如0號和1號)
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

### 8. 監控訓練（推薦）

雖然當前代碼沒有直接使用TensorBoard，但建議添加TensorBoard支持來更好地監控訓練：

```bash
# 首先在代碼中添加TensorBoard支援
# 然後運行TensorBoard服務
tensorboard --logdir=./results_lab6/runs
```

這樣，您可以在瀏覽器中通過 `http://localhost:6006` 訪問訓練監控面板。

### 注意事項

- 確保您的機器有足夠的GPU顯存（推薦至少8GB）
- 訓練過程可能需要較長時間，請確保環境穩定
- 如果顯存不足，可以嘗試減小批次大小（batch_size）或使用混合精度訓練