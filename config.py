# 所有参数集中管理，想改什么只动这里
import os

# ============ 路径 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "dataset", "aria-midi-v1-unique-ext", "data")
# metadata 位于 aria-midi-v1-unique-ext 根目录
METADATA_PATH = os.path.join(BASE_DIR, "dataset", "aria-midi-v1-unique-ext", "metadata.json")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# ============ 数据 ============
VELOCITY_NORMALIZE = 64       # 输入力度统一抹成这个值
MAX_NOTES = 512               # 每段最多音符数（超过截断，不足补零）
MIN_NOTES = 4                 # 过滤掉音符太少的曲子
TRAIN_RATIO = 0.9             # 训练集比例
TIME_TICK = 0.02              # 时间量化单位（秒），约 50fps

# ============ 模型 ============
D_MODEL = 256                 # 隐藏维度
N_LAYERS = 6                  # Transformer 层数
N_HEADS = 8                   # 注意力头数
DIM_FEEDFORWARD = 512         # 前馈网络维度
DROPOUT = 0.1
NUM_CLASSES = 128             # velocity 0-127，共 128 类

# ============ 训练 ============
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EPOCHS = 100
WARMUP_STEPS = 1000
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 8      # 验证集连续多少个 epoch 不提升就停止
EARLY_STOP_MIN_DELTA = 1e-3  # 认为“有提升”的最小验证损失变化

# ============ 硬件 ============
DEVICE = "cuda"               # 有 GPU 就用 "cuda"，没有就改 "cpu"
NUM_WORKERS = 8               # 数据加载线程数
USE_AMP = True                # 混合精度训练（省显存）
