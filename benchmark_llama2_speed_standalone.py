import argparse
import csv
import json
import math
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


def _stringify_torch_dtypes(obj, _seen=None):
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return
    _seen.add(id(obj))

    if hasattr(obj, "__dict__"):
        items = list(vars(obj).items())
    elif isinstance(obj, dict):
        items = list(obj.items())
    else:
        return

    for k, v in items:
        if isinstance(v, torch.dtype):
            new = str(v).replace("torch.", "")
            if isinstance(obj, dict):
                obj[k] = new
            else:
                setattr(obj, k, new)
        elif isinstance(v, dict):
            _stringify_torch_dtypes(v, _seen)
        elif hasattr(v, "__dict__") and not isinstance(v, (str, bytes, int, float, bool, list, tuple)):
            _stringify_torch_dtypes(v, _seen)
        elif isinstance(v, (list, tuple)):
            for item in v:
                _stringify_torch_dtypes(item, _seen)


def _is_lowrank_module(mod):
    if isinstance(mod, LowRankLinear):
        return True
    return (
        type(mod).__name__ == "LowRankLinear"
        and isinstance(getattr(mod, "v_proj", None), nn.Linear)
        and isinstance(getattr(mod, "u_proj", None), nn.Linear)
        and hasattr(mod, "in_features")
        and hasattr(mod, "out_features")
        and hasattr(mod, "rank")
    )


def collect_lowrank_specs(model):
    specs = {}

    for name, mod in model.named_modules():
        if _is_lowrank_module(mod):
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

    import transformers.activations as _hf_acts
    if not hasattr(_hf_acts, "SiLUActivation"):
        _hf_acts.SiLUActivation = nn.SiLU

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
    _stringify_torch_dtypes(model.config)
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


# ---------------------------------------------------------------------
# Phase 1: standalone low-rank V-cache generation path
# ---------------------------------------------------------------------
def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to transformers.models.llama.modeling_llama.repeat_kv.

    Input:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]

    Output:
        hidden_states: [batch, num_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _select_rope_positions(cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor):
    """
    Normalizes rotary embedding outputs from several LLaMA HF versions.

    Returns:
        cos_pos, sin_pos with shape [batch, 1, seq_len, head_dim]
    """
    if cos.dim() == 4:
        cos = cos.squeeze(0).squeeze(0)
        sin = sin.squeeze(0).squeeze(0)

    if cos.dim() == 2:
        cos = cos[position_ids]
        sin = sin[position_ids]
        return cos.unsqueeze(1), sin.unsqueeze(1)

    if cos.dim() == 3:
        return cos.unsqueeze(1), sin.unsqueeze(1)

    raise RuntimeError(f"Unsupported RoPE cos/sin shape: {tuple(cos.shape)}")


def _get_rope_cos_sin(rotary_emb, query_states, position_ids, kv_seq_len):
    """
    Calls a LLaMA rotary embedding in a version-tolerant way.

    Older transformers attach this to self_attn; newer versions hang it
    off LlamaModel. Caller is responsible for finding it.
    """
    if rotary_emb is None:
        raise RuntimeError(
            "This standalone low-rank V-cache path could not locate a "
            "LlamaRotaryEmbedding (checked self_attn.rotary_emb and model.model.rotary_emb)."
        )

    try:
        cos, sin = rotary_emb(query_states, position_ids)
        return cos, sin
    except TypeError:
        pass

    try:
        cos, sin = rotary_emb(query_states, seq_len=kv_seq_len)
        return cos, sin
    except TypeError:
        pass

    cos, sin = rotary_emb(query_states)
    return cos, sin


def _apply_rope_standalone(q, k, cos, sin, position_ids):
    cos, sin = _select_rope_positions(cos, sin, position_ids)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _validate_lowrank_v_cache_model(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("--lowrank_v_cache currently expects a LLaMA-style model.")

    n_checked = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        if not isinstance(attn.v_proj, LowRankLinear):
            raise RuntimeError(
                "--lowrank_v_cache requires every self_attn.v_proj to be LowRankLinear. "
                "Use --model_kind lowrank with a checkpoint whose v_proj modules were "
                "low-rank decomposed."
            )
        n_checked += 1

    print(f"[lowrank-v-cache] validated {n_checked} attention layers")


def _lowrank_v_attention_forward(
    attn,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key: torch.Tensor | None,
    past_z_value: torch.Tensor | None,
    rotary_emb=None,
):
    """
    LLaMA-style attention with:

        full K cache:
            key_cache: [batch, num_kv_heads, total_seq, head_dim]

        compressed V cache:
            z_value_cache: [batch, total_seq, rank_v]

    This is exact for the low-rank model when v_proj = u_proj(v_proj(x)).
    """
    if not isinstance(attn.v_proj, LowRankLinear):
        raise RuntimeError("Expected attn.v_proj to be LowRankLinear.")

    bsz, q_len, _ = hidden_states.shape
    device = hidden_states.device

    cfg = getattr(attn, "config", None)
    num_heads = int(
        getattr(attn, "num_heads", None)
        or (cfg.num_attention_heads if cfg is not None else None)
    )
    num_kv_heads = int(
        getattr(attn, "num_key_value_heads", None)
        or (cfg.num_key_value_heads if cfg is not None else num_heads)
    )
    num_kv_groups = int(
        getattr(attn, "num_key_value_groups", num_heads // num_kv_heads)
    )
    head_dim = int(
        getattr(attn, "head_dim", None)
        or (cfg.hidden_size // cfg.num_attention_heads if cfg is not None else None)
    )

    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    z_value_states = attn.v_proj.v_proj(hidden_states)

    past_len = 0 if past_key is None else past_key.shape[2]
    kv_seq_len = past_len + q_len

    rope = rotary_emb if rotary_emb is not None else getattr(attn, "rotary_emb", None)
    cos, sin = _get_rope_cos_sin(
        rotary_emb=rope,
        query_states=query_states,
        position_ids=position_ids,
        kv_seq_len=kv_seq_len,
    )
    query_states, key_states = _apply_rope_standalone(
        query_states,
        key_states,
        cos,
        sin,
        position_ids,
    )

    if past_key is not None:
        key_cache = torch.cat([past_key, key_states], dim=2)
        z_value_cache = torch.cat([past_z_value, z_value_states], dim=1)
    else:
        key_cache = key_states
        z_value_cache = z_value_states

    key_for_scores = _repeat_kv(key_cache, num_kv_groups)

    attn_scores = torch.matmul(query_states, key_for_scores.transpose(2, 3))
    attn_scores = attn_scores / math.sqrt(head_dim)

    key_positions = torch.arange(kv_seq_len, device=device).view(1, 1, 1, kv_seq_len)
    query_positions = position_ids.view(bsz, 1, q_len, 1)
    causal_mask = key_positions <= query_positions
    attn_scores = attn_scores.masked_fill(~causal_mask, torch.finfo(attn_scores.dtype).min)

    if attention_mask is not None:
        key_padding_mask = attention_mask[:, None, None, :kv_seq_len].to(torch.bool)
        attn_scores = attn_scores.masked_fill(
            ~key_padding_mask,
            torch.finfo(attn_scores.dtype).min,
        )

    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)

    z_context = torch.einsum("bhqs,bsr->bhqr", attn_probs, z_value_cache)

    u_weight = attn.v_proj.u_proj.weight
    rank_v = int(attn.v_proj.rank)
    expected_out = num_kv_heads * head_dim

    if u_weight.shape[0] != expected_out:
        raise RuntimeError(
            f"Unexpected v_proj.u_proj output dimension: got {u_weight.shape[0]}, "
            f"expected {expected_out} = num_kv_heads({num_kv_heads}) * head_dim({head_dim})."
        )

    u_blocks = u_weight.view(num_kv_heads, head_dim, rank_v)

    kv_head_for_query_head = torch.arange(num_heads, device=device) // num_kv_groups
    u_for_heads = u_blocks[kv_head_for_query_head]

    attn_output = torch.einsum("bhqr,hdr->bhqd", z_context, u_for_heads)

    if attn.v_proj.u_proj.bias is not None:
        bias_blocks = attn.v_proj.u_proj.bias.view(num_kv_heads, head_dim)
        bias_for_heads = bias_blocks[kv_head_for_query_head]
        attn_output = attn_output + bias_for_heads.view(1, num_heads, 1, head_dim)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    attn_output = attn.o_proj(attn_output)

    return attn_output, key_cache, z_value_cache


@torch.inference_mode()
def lowrank_v_cache_forward(model, input_ids, attention_mask, past_layer_caches=None):
    """
    Minimal standalone LLaMA forward pass using compressed V cache.

    past_layer_caches:
        None, or list of length num_layers.
        Each entry is a tuple:
            (key_cache, z_value_cache)
    """
    decoder = model.model
    bsz, q_len = input_ids.shape
    device = input_ids.device

    decoder_rotary_emb = getattr(decoder, "rotary_emb", None)

    past_len = 0
    if past_layer_caches is not None and past_layer_caches[0] is not None:
        past_len = past_layer_caches[0][0].shape[2]

    position_ids = torch.arange(
        past_len,
        past_len + q_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).expand(bsz, -1)

    hidden_states = decoder.embed_tokens(input_ids)
    new_layer_caches = []

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        past_key = None
        past_z_value = None

        if past_layer_caches is not None:
            past_key, past_z_value = past_layer_caches[layer_idx]

        residual = hidden_states
        hidden_states_norm = decoder_layer.input_layernorm(hidden_states)

        attn_output, new_key, new_z_value = _lowrank_v_attention_forward(
            attn=decoder_layer.self_attn,
            hidden_states=hidden_states_norm,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key=past_key,
            past_z_value=past_z_value,
            rotary_emb=decoder_rotary_emb,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        new_layer_caches.append((new_key, new_z_value))

    hidden_states = decoder.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return logits, new_layer_caches


@torch.inference_mode()
def lowrank_v_cache_generate(model, input_ids, attention_mask, max_new_tokens):
    """
    Greedy generation using the standalone compressed V-cache path.
    """
    generated = input_ids

    logits, layer_caches = lowrank_v_cache_forward(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_layer_caches=None,
    )

    for gen_idx in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
            dim=1,
        )

        if gen_idx + 1 >= max_new_tokens:
            break

        logits, layer_caches = lowrank_v_cache_forward(
            model=model,
            input_ids=next_token,
            attention_mask=attention_mask,
            past_layer_caches=layer_caches,
        )

    return generated


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


@torch.inference_mode()
def benchmark_generate_lowrank_v_cache(model, tokenizer, loader, device, gen_len, num_batches, warmup):
    if device.type != "cuda":
        raise RuntimeError("This benchmark script expects CUDA.")

    _validate_lowrank_v_cache_model(model)

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

        _ = lowrank_v_cache_generate(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_len,
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
    ap.add_argument(
        "--lowrank_v_cache",
        action="store_true",
        help=(
            "Use standalone phase-1 compressed V-cache generation. "
            "Requires --model_kind lowrank and LowRankLinear self_attn.v_proj modules."
        ),
    )

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

    if args.lowrank_v_cache and args.model_kind != "lowrank":
        raise ValueError("--lowrank_v_cache requires --model_kind lowrank")

    if args.lowrank_v_cache and args.compile:
        raise ValueError("--lowrank_v_cache currently does not support --compile")

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
    print("lowrank_v_cache:", args.lowrank_v_cache)
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
    if args.lowrank_v_cache:
        summary, batch_rows = benchmark_generate_lowrank_v_cache(
            model=model,
            tokenizer=tokenizer,
            loader=loader,
            device=device,
            gen_len=args.gen_len,
            num_batches=args.num_batches,
            warmup=args.warmup,
        )
    else:
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
        "kv_cache_impl": "lowrank_v_cache_phase1" if args.lowrank_v_cache else "hf_generate_default",
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
                "kv_cache_impl": "lowrank_v_cache_phase1" if args.lowrank_v_cache else "hf_generate_default",
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