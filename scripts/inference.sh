#!/bin/bash
set -x

MODEL_PATH=""
N=32

python inference_single.py \
    --model_path="${MODEL_PATH}" \
    --data_path='["../datasets/math_500.parquet", "../datasets/aime_2024.parquet", "../datasets/aime_2025.parquet","../datasets/amc23.parquet"]' \
    --n=${N} \
    --max_tokens=4096 \
    --temperature=0.6 \
    --top_p 0.95\
    --tensor_parallel_size 4 \
    2>&1 | tee ../logs/eval/eval.log
