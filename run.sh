#!/bin/bash
# DualParser：缓存 + Drain + 三解析器投票 + 大模型修复
# 无标注：不需要真值，也不需要训练任何模型。
set -e

datasets=("Hadoop" "HDFS" "OpenStack" "Spark" "Zookeeper" "BGL" "HPC" \
          "Thunderbird" "Linux" "Mac" "Apache" "OpenSSH" "HealthApp" "Proxifier")

DATA_DIR=${DATA_DIR:-datasets}
OUT_DIR=${OUT_DIR:-result}
DATA_TYPE=${DATA_TYPE:-full}
MODEL=${MODEL:-deepseek-chat}

mkdir -p "$OUT_DIR"

for dataset in "${datasets[@]}"; do
    echo "=== $dataset ==="
    python3 main.py \
        --dataset "$dataset" \
        --data_dir "$DATA_DIR" \
        --out_dir "$OUT_DIR" \
        --data_type "$DATA_TYPE" \
        --model "$MODEL" \
        --shot 0 \
        --eval
done

echo "All datasets completed."
