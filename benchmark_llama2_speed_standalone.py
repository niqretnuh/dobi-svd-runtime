import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


# ---------------------------------------------------------------------
# Low-rank module
# ---------------------------------------------------------------------
class LowRankLinear(nn.Module):
    """
    Drop-in low-rank replacement for nn.Linear.

    Dense:
        y = x W^T

    Low-rank:
        y = u_proj(v_proj(x))
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        self.v_proj = nn.Linear(self.in_features, self.rank, bias=False)
        self.u_proj = nn.Linear(self.rank, self.out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u_proj(self.v_proj(x))


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------
def exported_model_exists(export_dir: str) -> bool:
    export_dir = Path(export_dir)
    return (
        (export_dir / "config.json").exists()
        and (export_dir / "model.safetensors").exists()
        and (export_dir / "lowrank_specs.json").exists()
    )


def collect_lowrank_specs(model):
    specs = {}

    for name, mod in model.named_modules():
        if isinstance(mod, LowRankLinear):
            specs[name] = {
                "in_features": int(mod.in_features),
                "out_features": int(mod.out_features),
                "rank": int(mod.rank),
                "bias": mod.u_proj.bias is not None,
            }

    return specs


def export_lowrank_checkpoint(pt_path: str, out_dir: str, tokenizer_name: str, force_export: bool = False):
    if exported_model_exists(out_dir) and not force_export:
        print(f"[export] already exists, skipping: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("[export] loading trusted pickle checkpoint")
    print("[export] pt_path:", pt_path)

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError("Expected checkpoint to be a dict containing key 'model'.")

    model = ckpt["model"]
    print("[export] model type:", type(model))

    specs = collect_lowrank_specs(model)
    print("[export] LowRankLinear modules:", len(specs))

    ranks = [v["rank"] for v in specs.values()]
    if ranks:
        print(
            "[export] rank stats:",
            "min", min(ranks),
            "mean", sum(ranks) / len(ranks),
            "max", max(ranks),
        )

    print("[export] saving config")
    model.config.save_pretrained(out_dir)

    # Important: do NOT reuse ckpt["tokenizer"], because the pickled tokenizer may be stale.
    print("[export] saving fresh tokenizer from:", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(out_dir)

    specs_path = os.path.join(out_dir, "lowrank_specs.json")
    with open(specs_path, "w") as f:
        json.dump(specs, f, indent=2)

    print("[export] saving safetensors")
    sd = model.state_dict()
    sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))

    print("[export] complete:", out_dir)


# ---------------------------------------------------------------------
# Model reconstruction utilities
# ---------------------------------------------------------------------
def get_module(root, path):
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def set_module(root, path, new_mod):
    parent_path, leaf = path.rsplit(".", 1)
    parent = get_module(root, parent_path)
    setattr(parent, leaf, new_mod)


def apply_lowrank_specs(model, specs, dtype):
    for name, spec in specs.items():
        new_mod = LowRankLinear(
            in_features=spec["in_features"],
            out_features=spec["out_features"],
            rank=spec["rank"],
            bias=spec.get("bias", False),
        ).to(dtype=dtype)

        set_module(model, name, new_mod)

    ranks = [int(s["rank"]) for s in specs.values()]
    print(f"[lowrank] replaced {len(specs)} modules")

    if ranks:
        print(
            f"[lowrank] rank min/mean/max = "
            f"{min(ranks)}/{sum(ranks) / len(ranks):.2f}/{max(ranks)}"
        )


def reset_generation_config(model):
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


def load_dense_model(model_name_or_path, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    reset_generation_config(model)
    model = model.eval().to(device)

    return model, tokenizer


def load_lowrank_model(model_dir, dtype, device):
    specs_path = os.path.join(model_dir, "lowrank_specs.json")
    state_path = os.path.join(model_dir, "model.safetensors")

    if not os.path.exists(specs_path):
        raise FileNotFoundError(specs_path)
    if not os.path.exists(state_path):
        raise FileNotFoundError(state_path)

    with open(specs_path, "r") as f:
        specs = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_dir)

    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=dtype,
    )

    apply_lowrank_specs(model, specs, dtype=dtype)

    sd = load_file(state_path, device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)

    if missing or unexpected:
        raise RuntimeError(
            f"load_state_dict mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    reset_generation_config(model)
    model = model.eval().to(device)

    return model, tokenizer


# ---------------------------------------------------------------------
# Dataset / benchmark utilities
# ---------------------------------------------------------------------
def build_fixed_token_loader(tokenizer, prompt_len, batch_size, num_batches, warmup, max_rows):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    ids = []
    n = min(max_rows, len(ds))

    for row in ds.select(range(n)):
        text = row["text"]
        if not text or not text.strip():
            continue
        ids.extend(tokenizer(text, add_special_tokens=False).input_ids)

    needed_batches = warmup + num_batches
    needed_chunks = batch_size * needed_batches
    needed_tokens = needed_chunks * prompt_len

    if len(ids) < needed_tokens:
        raise RuntimeError(
            f"Not enough tokens for benchmark: have {len(ids)}, need {needed_tokens}. "
            f"Try smaller --batch_size, --prompt_len, --num_batches, or larger --max_rows."
        )

    chunks = []
    for i in range(needed_chunks):
        start = i * prompt_len
        chunks.append(ids[start:start + prompt_len])

    x = torch.tensor(chunks, dtype=torch.long)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False, drop_last=True)


@torch.inference_mode()
def benchmark_generate(model, tokenizer, loader, device, gen_len, num_batches, warmup):
    if device.type != "cuda":
        raise RuntimeError("This benchmark script expects CUDA.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    weight_mem = torch.cuda.memory_allocated(device)

    total_time = 0.0
    total_new_tokens = 0
    rows = []

    for step, (input_ids,) in enumerate(loader):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = torch.ones_like(input_ids, device=device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_len,
            min_new_tokens=gen_len,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,
        )

        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        if step < warmup:
            print(f"[warmup {step + 1}/{warmup}] {dt:.4f}s")
            continue

        measured_idx = step - warmup + 1
        new_tokens = input_ids.shape[0] * gen_len
        tok_s = new_tokens / dt

        total_time += dt
        total_new_tokens += new_tokens

        rows.append(
            {
                "batch_idx": measured_idx,
                "seconds": dt,
                "new_tokens": new_tokens,
                "new_tokens_per_sec": tok_s,
            }
        )

        print(f"[batch {measured_idx}/{num_batches}] {dt:.4f}s | {tok_s:.2f} new tok/s")

        if measured_idx >= num_batches:
            break

    peak_mem = torch.cuda.max_memory_allocated(device)

    summary = {
        "throughput_new_tokens_per_sec": total_new_tokens / total_time,
        "total_measured_seconds": total_time,
        "total_new_tokens": total_new_tokens,
        "weight_mem_gb": weight_mem / 2**30,
        "peak_mem_gb": peak_mem / 2**30,
        "activation_cache_mem_gb": (peak_mem - weight_mem) / 2**30,
    }

    return summary, rows


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Standalone export+benchmark script for dense Llama-2 or "
            "low-rank compressed Llama-2 checkpoints."
        )
    )

    # Which model to run
    ap.add_argument("--model_kind", choices=["dense", "lowrank"], required=True)

    # Dense path
    ap.add_argument("--dense_model", default="meta-llama/Llama-2-7b-hf")

    # Low-rank export inputs
    ap.add_argument("--pt_path", default="")
    ap.add_argument("--export_dir", default="")
    ap.add_argument("--tokenizer_name", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--force_export", action="store_true")
    ap.add_argument("--skip_export", action="store_true")

    # Benchmark metadata
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--seed", type=int, default=0)

    # Benchmark settings
    ap.add_argument("--prompt_len", type=int, default=1024)
    ap.add_argument("--gen_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_batches", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max_rows", type=int, default=20000)

    # Runtime options
    ap.add_argument("--compile", action="store_true")

    # Output
    ap.add_argument(
        "--out_csv",
        default="/home/qinh3/attention-aware-svdllm/clean_bench/results/export_benchmark_results.csv",
    )

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if device.type != "cuda":
        raise RuntimeError("CUDA device required for this benchmark.")

    free, total = torch.cuda.mem_get_info(device)
    print(f"[preflight] CUDA free={free / 2**30:.2f} GB / total={total / 2**30:.2f} GB")

    print("=" * 80)
    print("label:", args.label)
    print("model_kind:", args.model_kind)
    print("dense_model:", args.dense_model)
    print("pt_path:", args.pt_path)
    print("export_dir:", args.export_dir)
    print("device:", device)
    print("dtype:", dtype)
    print("compile:", args.compile)
    print("prompt_len:", args.prompt_len)
    print("gen_len:", args.gen_len)
    print("batch_size:", args.batch_size)
    print("=" * 80)

    # 1. Export if low-rank
    if args.model_kind == "lowrank":
        if not args.export_dir:
            raise ValueError("--export_dir is required when --model_kind lowrank")

        if not args.skip_export:
            if not args.pt_path:
                raise ValueError("--pt_path is required for lowrank export unless --skip_export is passed")

            export_lowrank_checkpoint(
                pt_path=args.pt_path,
                out_dir=args.export_dir,
                tokenizer_name=args.tokenizer_name,
                force_export=args.force_export,
            )
        else:
            print("[export] skipping export by user request")

    # 2. Load model
    if args.model_kind == "dense":
        model, tokenizer = load_dense_model(args.dense_model, dtype=dtype, device=device)
        model_path_for_csv = args.dense_model
    else:
        model, tokenizer = load_lowrank_model(args.export_dir, dtype=dtype, device=device)
        model_path_for_csv = args.export_dir

    # 3. Optional compile
    if args.compile:
        print("[compile] applying torch.compile(mode='reduce-overhead')")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    # 4. Build benchmark data
    loader = build_fixed_token_loader(
        tokenizer=tokenizer,
        prompt_len=args.prompt_len,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup=args.warmup,
        max_rows=args.max_rows,
    )

    # 5. Benchmark
    summary, batch_rows = benchmark_generate(
        model=model,
        tokenizer=tokenizer,
        loader=loader,
        device=device,
        gen_len=args.gen_len,
        num_batches=args.num_batches,
        warmup=args.warmup,
    )

    result_row = {
        "label": args.label,
        "model_kind": args.model_kind,
        "model_path": model_path_for_csv,
        "pt_path": args.pt_path,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "compile": args.compile,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "warmup": args.warmup,
        **summary,
    }

    append_csv(args.out_csv, result_row)

    batch_csv = args.out_csv.replace(".csv", "_per_batch.csv")
    for br in batch_rows:
        append_csv(
            batch_csv,
            {
                "label": args.label,
                "compile": args.compile,
                "prompt_len": args.prompt_len,
                "gen_len": args.gen_len,
                "batch_size": args.batch_size,
                **br,
            },
        )

    print("\nSUMMARY")
    for k, v in result_row.items():
        print(f"{k}: {v}")

    print("\nwrote:")
    print(args.out_csv)
    print(batch_csv)


if __name__ == "__main__":
    main()