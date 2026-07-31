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
L = 16                      # TPS lead time: 每块故障盘在训练时生成 l=1..L 条正样本
TEST_LEAD_TIME = 7          # 测试时不扩增，使用固定 lead time 天数

# ========== 数据维度 ==========
FEAT_DIM = 30            # SMART特征维度（原51，剔除20列100%NaN + 1列73.2%NaN，保留30列能被NUM_HEADS整除）
SEQ_LEN = 15             # 时间步长 h（15天窗口，捕获时序退化）
MAX_NEIGHBORS = 5        # 最大邻居数量 M

# ========== 模型结构 ==========
TRANSFORMER_LAYERS = 10   # Transformer编码器层数（提升模型容量）
NUM_HEADS =  3           # 多头注意力头数（30/3=10，整除）
DROPOUT = 0.1            # Dropout比率

# ========== 邻域组件开关 ==========
USE_NEIGHBORHOOD = True   # True=完整NTAM, False=消融实验(无邻域组件, 对应论文 NTAM_alt1)

# ========== 训练 ==========
BATCH_SIZE = 64          # 批次大小
LEARNING_RATE = 1e-3
EPOCHS = 5              # 增加训练轮次（10→20），给模型更多机会学习正样本模式
POS_WEIGHT = 0           # 不使用正样本权重（0=关闭）
VAL_SPLIT = 0.1          # 验证集比例（10%训练样本做早停）
PATIENCE = 5             # 早停耐心值：验证Loss连续5个epoch不降则停止
TRAIN_SHARDS = 50        # 恢复 10 分片（样本量增大）
TEST_SHARDS = 50         # 测试样本增多，增加分片数降低单片内存
MAX_TEST_SAMPLES = -1    # 测试集最大样本数（超出时随机采样，保持原始分布比例）
USE_AMP = False          # 关闭混合精度
MEMORY_LIMIT_GB = 8      # 内存看门狗阈值（GB），超过此值自动终止进程防止死机
DEVICE = "cuda"
