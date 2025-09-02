#!/bin/bash
set -ux

python preprocess_dataset.py \
    --input_path ../datasets/aime_2024/aime_2024_problems.parquet\
    --repeat 1

python preprocess_dataset.py \
    --input_path ../datasets/aime_2025/train-00000-of-00001-243207c6c994e1bd.parquet\
    --repeat 1

python preprocess_dataset.py \
    --input_path ../datasets/amc23/test-00000-of-00001.parquet\
    --repeat 1