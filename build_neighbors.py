# build_neighbors.py
# 从 target_disks.csv 读取目标磁盘清单，按 node_id 构建邻居关系并保存为长表格式
#
# 输入：
#   datasets/processed/target_disks.csv  (由 prefilter.py 生成)
# 输出：
#   datasets/processed/neighbor_map.csv  长表格式，列：pair_id, neighbor_pair_id, node_id
#
# 特点：
#   - 邻居关系仅由 node_id 决定（同 node 即邻居），与时间窗口、训练/测试划分无关
#   - 保存全部邻居（不截断），MAX_NEIGHBORS 截断在模型训练时按需进行
#   - 修改 MAX_NEIGHBORS 后无需重新运行本脚本
# 假设服务器 `node_id=42` 上有 3 块盘：`A`、`B`、`C`。生成的记录为：

# ```javascript
# pair_id, neighbor_pair_id, node_id
# A,      B,                42
# A,      C,                42
# B,      A,                42
# B,      C,                42
# C,      A,                42
# C,      B,                42
# ```
# 每行 = 一条 __有向邻接关系__。A 的邻居是 B 和 C，所以 A 有两行。


import os
import sys
import pandas as pd
from collections import defaultdict

# ========== 参数 ==========
TARGET_PATH = os.path.join("datasets", "processed", "target_disks.csv")
NEIGHBOR_PATH = os.path.join("datasets", "processed", "neighbor_map.csv")


def main():
    print("=" * 60)
    print("[build_neighbors] 从 target_disks.csv 构建邻居映射")

    # 1. 读取目标磁盘清单
    if not os.path.exists(TARGET_PATH):
        raise FileNotFoundError(f"目标磁盘清单不存在: {TARGET_PATH}\n请先运行 prefilter.py 生成该文件")

    print(f"\n[1/3] 读取 {TARGET_PATH}...")
    target_df = pd.read_csv(TARGET_PATH)
    n_total = len(target_df)
    print(f"  磁盘总数: {n_total:,}")

    # 2. 按 node_id 分组
    print(f"\n[2/3] 按 node_id 构建邻居关系 (同 node 即邻居)...")

    # 过滤有效的 node_id（>= 0）
    valid = target_df[target_df['node_id'] >= 0]
    n_valid = len(valid)
    n_invalid = n_total - n_valid
    print(f"  有效 node_id 的磁盘: {n_valid:,} | 无效 node_id: {n_invalid:,}")

    # 按 node_id 分组
    node_to_pairs = defaultdict(list)
    for _, row in valid.iterrows():
        node_to_pairs[int(row['node_id'])].append(row['pair_id'])

    n_nodes = len(node_to_pairs)
    print(f"  唯一 node 数: {n_nodes:,}")

    # 3. 生成长表
    rows = []
    n_pairs_with_neighbors = 0
    max_neighbors = 0
    n_total_relations = 0

    for node_id, pair_ids in node_to_pairs.items():
        for pid in pair_ids:
            neighbors = [n for n in pair_ids if n != pid]
            if len(neighbors) > 0:
                n_pairs_with_neighbors += 1
            max_neighbors = max(max_neighbors, len(neighbors))
            for neighbor_pid in neighbors:
                rows.append({
                    'pair_id': pid,
                    'neighbor_pair_id': neighbor_pid,
                    'node_id': node_id
                })
                n_total_relations += 1

    # 4. 保存
    print(f"\n[3/3] 保存邻居映射...")
    neighbor_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(NEIGHBOR_PATH), exist_ok=True)
    neighbor_df.to_csv(NEIGHBOR_PATH, index=False)

    # 统计摘要
    print(f"\n{'=' * 60}")
    print(f"  构建完成！")
    print(f"  有效磁盘: {n_valid:,} (有 node_id)")
    print(f"  至少有一个邻居的磁盘: {n_pairs_with_neighbors:,}")
    print(f"  无邻居的磁盘 (node 内仅自己): {n_valid - n_pairs_with_neighbors:,}")
    print(f"  最大邻居数: {max_neighbors}")
    print(f"  邻居关系总行数: {n_total_relations:,}")
    print(f"  → 已保存: {NEIGHBOR_PATH}")
    print(f"  → 文件大小: {os.path.getsize(NEIGHBOR_PATH) / 1024**2:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()