#!/usr/bin/env bash
set -euo pipefail

conda activate llama2-speed

cd /home/qinh3/attention-aware-svdllm/clean_bench

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_CSV="/home/qinh3/attention-aware-svdllm/clean_bench/results/standalone_grid_diagonal_b8.csv"

LENGTHS=(128 256 512 1024 2048)

for L in "${LENGTHS[@]}"; do
  echo "======================================================================"
  echo "Diagonal grid length: prompt=${L}, gen=${L}, batch=8"
  echo "======================================================================"

  # Dense / full Llama-2
  python export_and_benchmark_one.py \
    --model_kind dense \
    --dense_model meta-llama/Llama-2-7b-hf \
    --label dense_p${L}_g${L}_b8 \
    --device cuda \
    --prompt_len "${L}" \
    --gen_len "${L}" \
    --batch_size 8 \
    --num_batches 3 \
    --warmup 2 \
    --dtype fp16 \
    --out_csv "${OUT_CSV}" || true

  # Low-rank prune 0.4
  python export_and_benchmark_one.py \
    --model_kind lowrank \
    --pt_path /home/qinh3/attention-aware-svdllm/results/compressed_model_prune_rate_0.4/after_truncation_fp16_ppl10p96_20260417_163809.pt \
    --export_dir /home/qinh3/attention-aware-svdllm/exported/llama2_prune_rate_0.4 \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --label lowrank_0p4_p${L}_g${L}_b8 \
    --device cuda \
    --prompt_len "${L}" \
    --gen_len "${L}" \
    --batch_size 8 \
    --num_batches 3 \
    --warmup 2 \
    --dtype fp16 \
    --out_csv "${OUT_CSV}" || true

  # Low-rank prune 0.6
  python export_and_benchmark_one.py \
    --model_kind lowrank \
    --pt_path /home/qinh3/attention-aware-svdllm/results/compressed_model_prune_rate_0.6/after_truncation_fp16_ppl34p91_20260417_163916.pt \
    --export_dir /home/qinh3/attention-aware-svdllm/exported/llama2_prune_rate_0.6 \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --label lowrank_0p6_p${L}_g${L}_b8 \
    --device cuda \
    --prompt_len "${L}" \
    --gen_len "${L}" \
    --batch_size 8 \
    --num_batches 3 \
    --warmup 2 \
    --dtype fp16 \
    --out_csv "${OUT_CSV}" || true
done