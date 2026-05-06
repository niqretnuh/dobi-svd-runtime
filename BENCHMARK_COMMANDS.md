# Benchmark commands — Llama-2-7B speed grid

Shared knobs: `prompt_len=1024`, `gen_len=1024`, `batch_size=8`, `num_batches=3`, `warmup=1`, `dtype=fp16`, `CUDA_VISIBLE_DEVICES=1`.

Compressed checkpoint: `./llama2-jwsvd/after_truncation_fp16_ppl34p91_20260417_163916.pt` (jwsvd, gpr0p6).

## 1. Uncompressed dense (HuggingFace Llama-2-7B)

```bash
CUDA_VISIBLE_DEVICES=1 \
python benchmark_llama2_speed_standalone.py \
  --model_kind dense \
  --dense_model meta-llama/Llama-2-7b-hf \
  --label dense_llama2_7b_p1024_g1024_b8 \
  --device cuda \
  --prompt_len 1024 --gen_len 1024 \
  --batch_size 8 --num_batches 3 --warmup 1 \
  --dtype fp16 \
  --out_dir ~/dobi-svd-runtime/results
```

## 2. Compressed weights, no KV cache compression (full K/V cache, HF generate)

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/abbasa2/attention-aware-svdllm \
python benchmark_llama2_speed_standalone.py \
  --model_kind lowrank \
  --pt_path ./llama2-jwsvd/after_truncation_fp16_ppl34p91_20260417_163916.pt \
  --export_dir ~/dobi-svd-runtime/exported/llama2_jwsvd_ppl34p91 \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --label lowrank_jwsvd_gpr0p6_ppl34p91_p1024_g1024_b8_nocache \
  --device cuda \
  --prompt_len 1024 --gen_len 1024 \
  --batch_size 8 --num_batches 3 --warmup 1 \
  --dtype fp16 \
  --out_dir ~/dobi-svd-runtime/results
```

## 3. Compressed weights + V-only cache compression (preallocated)

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/abbasa2/attention-aware-svdllm \
python benchmark_llama2_speed_standalone.py \
  --model_kind lowrank \
  --pt_path ./llama2-jwsvd/after_truncation_fp16_ppl34p91_20260417_163916.pt \
  --export_dir ~/dobi-svd-runtime/exported/llama2_jwsvd_ppl34p91 \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --label lowrank_jwsvd_gpr0p6_ppl34p91_p1024_g1024_b8_vcache_prealloc \
  --device cuda \
  --prompt_len 1024 --gen_len 1024 \
  --batch_size 8 --num_batches 3 --warmup 1 \
  --dtype fp16 \
  --lowrank_v_cache --vcache_prealloc \
  --out_dir ~/dobi-svd-runtime/results
```

## 4. Compressed weights + K+V cache compression (preallocated, chunk=2048)

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/abbasa2/attention-aware-svdllm \
python benchmark_llama2_speed_standalone.py \
  --model_kind lowrank \
  --pt_path ./llama2-jwsvd/after_truncation_fp16_ppl34p91_20260417_163916.pt \
  --export_dir ~/dobi-svd-runtime/exported/llama2_jwsvd_ppl34p91 \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --label lowrank_jwsvd_gpr0p6_ppl34p91_p1024_g1024_b8_kvcache_prealloc_chunk2048 \
  --device cuda \
  --prompt_len 1024 --gen_len 1024 \
  --batch_size 8 --num_batches 3 --warmup 1 \
  --dtype fp16 \
  --lowrank_k_cache --kvcache_prealloc --k_score_chunk_size 2048 \
  --out_dir ~/dobi-svd-runtime/results
```
