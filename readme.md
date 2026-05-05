# Environment setup
conda create -n llama2-speed python=3.10 -y
conda activate llama2-speed

pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install \
  "transformers>=4.44,<4.47" \
  "accelerate>=0.33" \
  "datasets>=2.20" \
  "tokenizers>=0.19" \
  "safetensors>=0.4" \
  "sentencepiece" \
  "protobuf" \
  "numpy" \
  "pandas" \
  "tqdm"

# Benchmark for runtime for in/out length 1024
python benchmark_llama2_speed_standalone.py \
  --model_kind dense \
  --dense_model meta-llama/Llama-2-7b-hf \
  --label dense_llama2_7b_p1024_g1024_b8 \
  --device cuda \
  --prompt_len 1024 \
  --gen_len 1024 \
  --batch_size 8 \
  --num_batches 3 \
  --warmup 1 \
  --dtype fp16 \
  --out_csv /home/qinh3/attention-aware-svdllm/clean_bench/results/standalone_1024_b8.csv

python benchmark_llama2_speed_standalone.py \
  --model_kind lowrank \
  --pt_path /home/qinh3/attention-aware-svdllm/results/compressed_model_prune_rate_0.4/after_truncation_fp16_ppl10p96_20260417_163809.pt \
  --export_dir /home/qinh3/attention-aware-svdllm/exported/llama2_prune_rate_0.4 \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --label lowrank_0p4_p1024_g1024_b8 \
  --device cuda \
  --prompt_len 1024 \
  --gen_len 1024 \
  --batch_size 8 \
  --num_batches 3 \
  --warmup 1 \
  --dtype fp16 \
  --out_csv /home/qinh3/attention-aware-svdllm/clean_bench/results/standalone_1024_b8.csv

python benchmark_llama2_speed_standalone.py \
  --model_kind lowrank \
  --pt_path /home/qinh3/attention-aware-svdllm/results/compressed_model_prune_rate_0.6/after_truncation_fp16_ppl34p91_20260417_163916.pt \
  --export_dir /home/qinh3/attention-aware-svdllm/exported/llama2_prune_rate_0.6 \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --label lowrank_0p6_p1024_g1024_b8 \
  --device cuda \
  --prompt_len 1024 \
  --gen_len 1024 \
  --batch_size 8 \
  --num_batches 3 \
  --warmup 1 \
  --dtype fp16 \
  --out_csv /home/qinh3/attention-aware-svdllm/clean_bench/results/standalone_1024_b8.csv

# If we want to run a grid for prompt length vs. speed
chmod +x run_llama2_speed_grid.sh
./run_llama2_speed_grid.sh

