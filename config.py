# config.py
import torch
"""
分片管理方式（手动）：
  - 改任何 config 参数都不会自动重建分片。
  - 需要重建时，手动删除 datasets/processed 下对应的分片文件：
      train_shard_*.npz  → 训练分片
      test_shard_*.npz   → 测试分片
  - 删除后再次运行 train.py 会自动重新构建缺失的分片。
  - 分片数量由 TRAIN_SHARDS / TEST_SHARDS 决定（均匀切分）。
  - 改 TRAIN_START / TRAIN_END 后需先运行 python code/build_feat_r.py 重建特征。
"""
# ========== 数据范围 ==========
DATA_MONTH_START = 1     # 起始月份（1~12）
DATA_MONTH_END = 12       # 结束月份（1~12，验证测试用1~2月数据）
# ========== 训练/测试时间范围（自由指定，含边界，格式 YYYYMMDD）==========
# 语义：窗口结束日在范围内即归属该集合；窗口内容允许回看更早历史
TRAIN_START = "20180101"   # 训练集窗口结束日开始日期（含）—— 最近3个月，对齐论文"近邻时间"协议
TRAIN_END   = "20181108"   # 训练集窗口结束日结束日期（含）
TEST_START  = "20181113"   # 测试集窗口结束日开始日期（含）
TEST_END    = "20181231"   # 测试集窗口结束日结束日期（含，测试全年改 20181231）
TRAIN_CUTOFF = TRAIN_END   # 兼容旧分析脚本（analyze_smart/compare_r_vs_n/visualize_fail_trend）
MAX_DISKS = 0                # 参与实验的磁盘上限（0=全量）；故障盘全保留，健康盘随机采样补齐


# ========== TPS (Temporal Progressive Sampling) ==========
L = 4                                                                                                                                                                                                                                 # TPS lead time: 每块故障盘在训练时生成 l=1..L 条正样本（NTAM论文最优值 L=16）
TEST_LEAD_TIME = 7          # 测试时不扩增，使用固定 lead time 天数

# ========== 数据维度 ==========
FEAT_DIM = 30            # SMART特征维度（原51，剔除20列100%NaN + 1列73.2%NaN，保留30列能被NUM_HEADS整除）
SEQ_LEN = 8              # 时间步长 h（8天窗口，信号分析显示故障信号集中在故障前~7天）
MAX_NEIGHBORS = 5        # 最大邻居数量 M

# ========== 模型结构 ==========
TRANSFORMER_LAYERS = 3  # Transformer编码器层数
NUM_HEADS =  3           # 多头注意力头数
DROPOUT = 0.1            # Dropout比率                                                                                                                                                                                  

# ========== 邻域组件开关 ==========
USE_NEIGHBORHOOD = True   # True=完整NTAM, False=消融实验(无邻域组件, 对应论文 NTAM_alt1)

# ========== 训练 ==========
BATCH_SIZE = 256          # 批次大小
LEARNING_RATE = 1e-4
EPOCHS = 5              # EPOCHS上限，早停会自动提前终止
POS_WEIGHT = 0           # 正样本损失权重（对齐NTAM论文：用TPS处理不平衡，不使用pos_weight）
USE_VALIDATION = False  # 已废弃：改为每epoch直接评估测试集，此参数不再被 train.py 使用
VAL_SPLIT = 0.1          # 验证集比例（10%训练样本做早停）
PATIENCE = 3             # 早停耐心值：验证Loss连续5个epoch不降则停止
TRAIN_SHARDS = 5        # 训练分片数量（改后需手动删除分片重建）
TEST_SHARDS = 5         # 测试分片数量（改后需手动删除分片重建）
# 自定义选择加载哪些分片（做对照实验用，如"只丢第一片/只丢最后一片"）
# 空列表 = 加载前 N 个分片（默认行为）；指定如 TRAIN_SHARD_IDS=[1,2,3,4] 表示只加载这些分片
TRAIN_SHARD_IDS = []     # 训练分片 ID 列表（空=前 TRAIN_SHARDS 片）
TEST_SHARD_IDS = []      # 测试分片 ID 列表（空=前 TEST_SHARDS 片）
# 训练分片损失权重（让模型更偏重学习某些分片，如让模型多学训练集后面/接近测试期的分片）
# 空 = 全部等权 1.0；dict 按分片 ID 指定：{4: 2.0, 3: 1.5}；list 按分片 ID 顺序：[1.0, 1.0, 1.2, 1.5, 2.0]
TRAIN_SHARD_WEIGHTS = {} # 空=全部等权 1.0


MAX_TEST_SAMPLES = -1    # 测试集最大样本数（超出时随机采样，保持原始分布比例）
USE_AMP = False          # 关闭混合精度
MEMORY_LIMIT_GB = 8      # 内存看门狗阈值（GB），超过此值自动终止进程防止死机
# ========== 预测/判定阈值 ==========
PRED_THRESHOLD = 0.5   # 预测概率大于该阈值才判定为故障（调低→召回↑误报↑；调高→精确率↑漏报↑）
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 评估脚本扫描多阈值用（evaluate_model.py / _eval_report.py）

DEVICE = "cuda"

# ========== 微调（Fine-tune，复用训练集最后 N 个分片，无需重建数据）==========
FINE_TUNE_LAST_SHARDS = 1            # 取最后 N 个训练分片微调（分片按时间排序，越靠后越接近测试期）
FINE_TUNE_SHARD_WEIGHTS = [2.0] # 分片级时间权重（从旧到新，与 LAST_SHARDS 一一对应；最后一片权重最高=最偏近期）
FINE_TUNE_EPOCHS = 2                 # 微调 epoch（小，防遗忘）
FINE_TUNE_LR = 1e-6                  # 微调学习率（主训练 9e-5 的约 1/10）
FINE_TUNE_SAVE_PATH = "saved_models/ntam_finetuned.pt"  # 微调后模型保存路径（不覆盖原 best）

# ========== 模型保存 ==========
SAVE_DIR = "saved_models"  # 训练完成后保存最佳模型与训练日志的目录（相对当前工作目录）
