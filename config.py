# config.py
import torch
"""
改了 `TRAIN_CUTOFF`、`L`、`TEST_LEAD_TIME` 等参数后需&#
x8981;__&#x624B;动删除旧分&#x7247;__&#x518D;
重跑，否则会复用旧分片。`feat_day` 不需要删——它只和日期范围
 + MAX_NEIGHBORS 相关。
"""
# ========== 数据范围 ==========
DATA_MONTH_START = 1     # 起始月份（1~12）
DATA_MONTH_END = 12       # 结束月份（1~12，验证测试用1~2月数据）
TRAIN_CUTOFF = "20181115"  # 训练/测试分割日期（<=此日期为训练集，>为测试集）
MAX_DISKS = 0                # 参与实验的磁盘上限（0=全量）；故障盘全保留，健康盘随机采样补齐


# ========== TPS (Temporal Progressive Sampling) ==========
L = 5                       # TPS lead time: 每块故障盘在训练时生成 l=1..L 条正样本（L=4 优于 L=3，试 L=5 确认是否更优）
TEST_LEAD_TIME = 7          # 测试时不扩增，使用固定 lead time 天数

# ========== 数据维度 ==========
FEAT_DIM = 30            # SMART特征维度（原51，剔除20列100%NaN + 1列73.2%NaN，保留30列能被NUM_HEADS整除）
SEQ_LEN = 8              # 时间步长 h（8天窗口，信号分析显示故障信号集中在故障前~7天）
MAX_NEIGHBORS = 5        # 最大邻居数量 M

# ========== 模型结构 ==========
TRANSFORMER_LAYERS = 3  # Transformer编码器层数（最佳配置：10层）
NUM_HEADS =  3           # 多头注意力头数（30/3=10，整除）
DROPOUT = 0.1            # Dropout比率

# ========== 邻域组件开关 ==========
USE_NEIGHBORHOOD = True   # True=完整NTAM, False=消融实验(无邻域组件, 对应论文 NTAM_alt1)

# ========== 训练 ==========
BATCH_SIZE = 64          # 批次大小
LEARNING_RATE = 9e-5
EPOCHS = 5              # EPOCHS上限，早停会自动提前终止
POS_WEIGHT = 0           # 不使用正样本权重（0=关闭）
USE_VALIDATION = False  # 已废弃：改为每epoch直接评估测试集，此参数不再被 train.py 使用
VAL_SPLIT = 0.1          # 验证集比例（10%训练样本做早停）
PATIENCE = 3             # 早停耐心值：验证Loss连续5个epoch不降则停止
TRAIN_SHARDS = 10        # 恢复 10 分片（样本量增大）
TEST_SHARDS = 10         # 测试样本增多，增加分片数降低单片内存
MAX_TEST_SAMPLES = -1    # 测试集最大样本数（超出时随机采样，保持原始分布比例）
USE_AMP = False          # 关闭混合精度
MEMORY_LIMIT_GB = 8      # 内存看门狗阈值（GB），超过此值自动终止进程防止死机
DEVICE = "cuda"

# ========== 模型保存 ==========
SAVE_DIR = "saved_models"  # 训练完成后保存最佳模型与训练日志的目录（相对当前工作目录）
