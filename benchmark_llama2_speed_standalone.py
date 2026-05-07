import argparse
import json
import math
import os
import random
import re
import shlex
import sys
import time
from datetime import datetime
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


def _apply_rope_to_q_standalone(q, cos, sin, position_ids):
    cos, sin = _select_rope_positions(cos, sin, position_ids)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    return q_embed


def _apply_rope_to_k_standalone(k, cos, sin, position_ids):
    cos, sin = _select_rope_positions(cos, sin, position_ids)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return k_embed


def _get_llama_attn_dims(attn):
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

    return num_heads, num_kv_heads, num_kv_groups, head_dim


def _cache_seq_len(cache):
    if cache is None:
        return 0

    if cache.dim() == 4:
        return int(cache.shape[2])

    if cache.dim() == 3:
        return int(cache.shape[1])

    raise RuntimeError(f"Unsupported cache shape: {tuple(cache.shape)}")


def _bytes_to_gb(x) -> float:
    return float(x) / float(2**30)


def _unwrap_compiled_model(model):
    """torch.compile wraps the original module under _orig_mod."""
    return getattr(model, "_orig_mod", model)


def _infer_cache_element_size(model) -> int:
    """Returns bytes per cache element (cache dtype follows weight dtype)."""
    model = _unwrap_compiled_model(model)

    for p in model.parameters():
        return p.element_size()

    raise RuntimeError("Could not infer model/cache dtype because model has no parameters.")


def _full_k_or_v_cache_bytes(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    elem_size: int,
) -> int:
    return int(batch_size) * int(seq_len) * int(num_kv_heads) * int(head_dim) * int(elem_size)


def _lowrank_cache_bytes(
    batch_size: int,
    seq_len: int,
    rank: int,
    elem_size: int,
) -> int:
    return int(batch_size) * int(seq_len) * int(rank) * int(elem_size)


def estimate_kv_cache_bytes(
    model,
    batch_size: int,
    seq_len: int,
    cache_kind: str,
) -> int:
    """
    Estimates persistent KV-cache footprint for the selected cache mode.

    cache_kind:
        "hf_full"     full K + full V per layer
        "lowrank_v"   full K, V compressed if v_proj is LowRankLinear else full V
        "lowrank_kv"  K compressed if k_proj is LowRankLinear else full K
                      V compressed if v_proj is LowRankLinear else full V

    Returns the steady-state cache size at seq_len = prompt_len + gen_len.
    Excludes attention scores, softmax, logits, torch.cat reallocation, and
    allocator fragmentation; those land in other_runtime_mem_gb.
    """
    model = _unwrap_compiled_model(model)

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("KV-cache memory estimation currently expects a LLaMA-style model.")

    if cache_kind not in {"hf_full", "lowrank_v", "lowrank_kv"}:
        raise ValueError(f"Unsupported cache_kind: {cache_kind}")

    elem_size = _infer_cache_element_size(model)
    total_bytes = 0

    for layer in model.model.layers:
        attn = layer.self_attn
        _, num_kv_heads, _, head_dim = _get_llama_attn_dims(attn)

        full_one_side = _full_k_or_v_cache_bytes(
            batch_size=batch_size,
            seq_len=seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            elem_size=elem_size,
        )

        if cache_kind == "hf_full":
            total_bytes += full_one_side
            total_bytes += full_one_side
            continue

        if cache_kind == "lowrank_v":
            total_bytes += full_one_side
            if _is_lowrank_module(attn.v_proj):
                total_bytes += _lowrank_cache_bytes(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    rank=int(attn.v_proj.rank),
                    elem_size=elem_size,
                )
            else:
                total_bytes += full_one_side
            continue

        if cache_kind == "lowrank_kv":
            if _is_lowrank_module(attn.k_proj):
                total_bytes += _lowrank_cache_bytes(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    rank=int(attn.k_proj.rank),
                    elem_size=elem_size,
                )
            else:
                total_bytes += full_one_side

            if _is_lowrank_module(attn.v_proj):
                total_bytes += _lowrank_cache_bytes(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    rank=int(attn.v_proj.rank),
                    elem_size=elem_size,
                )
            else:
                total_bytes += full_one_side

    return int(total_bytes)


def build_memory_breakdown_summary(
    *,
    model,
    total_time: float,
    total_new_tokens: int,
    weight_mem_bytes: int,
    peak_mem_bytes: int,
    cache_batch_size: int,
    cache_seq_len: int,
    cache_kind: str,
) -> dict:
    """
    Builds the summary with a fine-grained memory split:
        weights / KV cache / other runtime memory.

    activation_cache_mem_gb is preserved as alias for non-weight peak memory.
    """
    kv_cache_bytes = estimate_kv_cache_bytes(
        model=model,
        batch_size=cache_batch_size,
        seq_len=cache_seq_len,
        cache_kind=cache_kind,
    )

    non_weight_bytes = max(0, int(peak_mem_bytes) - int(weight_mem_bytes))
    other_runtime_bytes_raw = int(peak_mem_bytes) - int(weight_mem_bytes) - int(kv_cache_bytes)
    other_runtime_bytes = max(0, other_runtime_bytes_raw)

    return {
        "throughput_new_tokens_per_sec": total_new_tokens / total_time,
        "total_measured_seconds": total_time,
        "total_new_tokens": total_new_tokens,

        "weight_mem_gb": _bytes_to_gb(weight_mem_bytes),
        "peak_mem_gb": _bytes_to_gb(peak_mem_bytes),
        "activation_cache_mem_gb": _bytes_to_gb(non_weight_bytes),

        "non_weight_runtime_mem_gb": _bytes_to_gb(non_weight_bytes),
        "kv_cache_mem_gb": _bytes_to_gb(kv_cache_bytes),
        "other_runtime_mem_gb": _bytes_to_gb(other_runtime_bytes),

        "weight_mem_bytes": int(weight_mem_bytes),
        "peak_mem_bytes": int(peak_mem_bytes),
        "non_weight_runtime_mem_bytes": int(non_weight_bytes),
        "kv_cache_mem_bytes": int(kv_cache_bytes),
        "other_runtime_mem_bytes": int(other_runtime_bytes),
        "other_runtime_mem_bytes_raw": int(other_runtime_bytes_raw),

        "memory_cache_kind": cache_kind,
        "memory_cache_batch_size": int(cache_batch_size),
        "memory_cache_seq_len": int(cache_seq_len),
    }


def _dense_kv_attention_forward(
    attn,
    hidden_states,
    position_ids,
    attention_mask,
    past_key,
    past_value,
    rotary_emb=None,
):
    bsz, q_len, _ = hidden_states.shape
    device = hidden_states.device

    num_heads, num_kv_heads, num_kv_groups, head_dim = _get_llama_attn_dims(attn)

    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    past_len = _cache_seq_len(past_key)
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
        value_cache = torch.cat([past_value, value_states], dim=2)
    else:
        key_cache = key_states
        value_cache = value_states

    key_for_scores = _repeat_kv(key_cache, num_kv_groups)
    value_for_context = _repeat_kv(value_cache, num_kv_groups)

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

    attn_output = torch.matmul(attn_probs, value_for_context)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    attn_output = attn.o_proj(attn_output)

    return attn_output, key_cache, value_cache


def _apply_inverse_rope_for_key_positions(q_rope, cos, sin, key_position_ids):
    """
    Applies R_j^T to a RoPE-rotated query for a chunk of key positions.

    q_rope:
        [batch, num_heads, q_len, head_dim]

    key_position_ids:
        [chunk]

    Returns:
        q_unrot:
            [batch, num_heads, q_len, chunk, head_dim]
    """
    key_position_ids = key_position_ids.view(1, -1)
    cos_key, sin_key = _select_rope_positions(cos, sin, key_position_ids)

    # [1, 1, chunk, head_dim] -> [1, 1, 1, chunk, head_dim]
    cos_key = cos_key.unsqueeze(2)
    sin_key = sin_key.unsqueeze(2)

    # [batch, heads, q_len, head_dim] -> [batch, heads, q_len, 1, head_dim]
    q_expanded = q_rope.unsqueeze(3)

    # Inverse RoPE: R_j^T x.
    return (q_expanded * cos_key) - (_rotate_half(q_expanded) * sin_key)


def _lowrank_k_scores_from_compressed_cache(
    q_rope: torch.Tensor,
    z_key_cache: torch.Tensor,
    u_for_heads: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    chunk_size: int,
):
    """
    Computes attention scores against a compressed low-rank K cache.

    q_rope:
        [batch, num_heads, q_len, head_dim]

    z_key_cache:
        [batch, total_seq, rank_k]

    u_for_heads:
        [num_heads, head_dim, rank_k]

    Returns:
        scores:
            [batch, num_heads, q_len, total_seq]
    """
    device = q_rope.device
    bsz, num_heads, q_len, _ = q_rope.shape
    total_seq = z_key_cache.shape[1]

    score_chunks = []

    for start in range(0, total_seq, chunk_size):
        end = min(start + chunk_size, total_seq)
        key_pos = torch.arange(start, end, device=device, dtype=torch.long)

        q_unrot = _apply_inverse_rope_for_key_positions(
            q_rope=q_rope,
            cos=cos,
            sin=sin,
            key_position_ids=key_pos,
        )

        # q_unrot:    [batch, heads, q_len, chunk, head_dim]
        # u_for_heads: [heads, head_dim, rank_k]
        # q_low:      [batch, heads, q_len, chunk, rank_k]
        q_low = torch.einsum("bhqsd,hdr->bhqsr", q_unrot, u_for_heads)

        z_chunk = z_key_cache[:, start:end, :]

        # scores_chunk: [batch, heads, q_len, chunk]
        scores_chunk = torch.einsum("bhqsr,bsr->bhqs", q_low, z_chunk)
        score_chunks.append(scores_chunk)

    scores = torch.cat(score_chunks, dim=-1)
    return scores / math.sqrt(head_dim)


def _validate_lowrank_v_cache_model(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("--lowrank_v_cache currently expects a LLaMA-style model.")

    n_layers = 0
    n_lowrank_v = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        n_layers += 1
        n_lowrank_v += int(_is_lowrank_module(attn.v_proj))

    print(
        f"[lowrank-v-cache] low-rank v_proj layers: "
        f"{n_lowrank_v}/{n_layers}; dense v_proj layers use full V cache"
    )


def _validate_lowrank_kv_cache_model(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("--lowrank_k_cache currently expects a LLaMA-style model.")

    n_layers = 0
    n_lowrank_k = 0
    n_lowrank_v = 0

    for layer in model.model.layers:
        attn = layer.self_attn

        n_layers += 1
        n_lowrank_k += int(_is_lowrank_module(attn.k_proj))
        n_lowrank_v += int(_is_lowrank_module(attn.v_proj))

    print(
        f"[lowrank-kv-cache] low-rank k_proj layers: "
        f"{n_lowrank_k}/{n_layers}; dense k_proj layers use full K cache"
    )
    print(
        f"[lowrank-kv-cache] low-rank v_proj layers: "
        f"{n_lowrank_v}/{n_layers}; dense v_proj layers use full V cache"
    )


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
    if not _is_lowrank_module(attn.v_proj):
        return _dense_kv_attention_forward(
            attn=attn,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key=past_key,
            past_value=past_z_value,
            rotary_emb=rotary_emb,
        )

    bsz, q_len, _ = hidden_states.shape
    device = hidden_states.device

    num_heads, num_kv_heads, num_kv_groups, head_dim = _get_llama_attn_dims(attn)

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


def _lowrank_kv_attention_forward(
    attn,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_z_key: torch.Tensor | None,
    past_z_value: torch.Tensor | None,
    rotary_emb=None,
    k_score_chunk_size: int = 256,
):
    """
    LLaMA-style attention with compressed K and compressed V cache.

    Cache format:
        z_key_cache:   [batch, total_seq, rank_k]
        z_value_cache: [batch, total_seq, rank_v]

    For prefill, this reconstructs the current full K inside the layer only.
    For decode, it computes scores directly from compressed K coefficients:

        score_ij = q_i^T R_i^T R_j U_k z_j
                 = (U_k^T R_j^T q_rope_i)^T z_j

    so the persistent K cache stays low-rank.
    """
    k_is_lowrank = _is_lowrank_module(attn.k_proj)
    v_is_lowrank = _is_lowrank_module(attn.v_proj)

    if not k_is_lowrank and not v_is_lowrank:
        return _dense_kv_attention_forward(
            attn=attn,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key=past_z_key,
            past_value=past_z_value,
            rotary_emb=rotary_emb,
        )

    if (past_z_key is None) != (past_z_value is None):
        raise RuntimeError("K and V caches must either both be None or both be present.")

    bsz, q_len, _ = hidden_states.shape
    device = hidden_states.device

    num_heads, num_kv_heads, num_kv_groups, head_dim = _get_llama_attn_dims(attn)

    query_states = attn.q_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    past_len = _cache_seq_len(past_z_key)
    kv_seq_len = past_len + q_len

    rope = rotary_emb if rotary_emb is not None else getattr(attn, "rotary_emb", None)
    cos, sin = _get_rope_cos_sin(
        rotary_emb=rope,
        query_states=query_states,
        position_ids=position_ids,
        kv_seq_len=kv_seq_len,
    )

    # K path. If k_proj is low-rank, store [batch, seq, rank_k].
    # Otherwise store full K as [batch, num_kv_heads, seq, head_dim].
    if k_is_lowrank:
        query_states = _apply_rope_to_q_standalone(
            query_states,
            cos,
            sin,
            position_ids,
        )

        z_key_states = attn.k_proj.v_proj(hidden_states)
        if past_z_key is not None:
            key_cache_out = torch.cat([past_z_key, z_key_states], dim=1)
        else:
            key_cache_out = z_key_states
    else:
        key_states = attn.k_proj(hidden_states)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        query_states, key_states = _apply_rope_standalone(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

        if past_z_key is not None:
            key_cache_out = torch.cat([past_z_key, key_states], dim=2)
        else:
            key_cache_out = key_states

    # V path. If v_proj is low-rank, store [batch, seq, rank_v].
    # Otherwise store full V as [batch, num_kv_heads, seq, head_dim].
    if v_is_lowrank:
        z_value_states = attn.v_proj.v_proj(hidden_states)
        if past_z_value is not None:
            value_cache_out = torch.cat([past_z_value, z_value_states], dim=1)
        else:
            value_cache_out = z_value_states
    else:
        value_states = attn.v_proj(hidden_states)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        if past_z_value is not None:
            value_cache_out = torch.cat([past_z_value, value_states], dim=2)
        else:
            value_cache_out = value_states

    kv_head_for_query_head = torch.arange(num_heads, device=device) // num_kv_groups

    if k_is_lowrank:
        k_u_weight = attn.k_proj.u_proj.weight
        rank_k = int(attn.k_proj.rank)
        expected_k_out = num_kv_heads * head_dim

        if k_u_weight.shape[0] != expected_k_out:
            raise RuntimeError(
                f"Unexpected k_proj.u_proj output dimension: got {k_u_weight.shape[0]}, "
                f"expected {expected_k_out} = num_kv_heads({num_kv_heads}) * head_dim({head_dim})."
            )

        k_u_blocks = k_u_weight.view(num_kv_heads, head_dim, rank_k)
        k_u_for_heads = k_u_blocks[kv_head_for_query_head]

    if k_is_lowrank and past_z_key is None:
        # Prefill path: reconstruct K only inside this layer for the prompt pass.
        # The persistent cache still stores low-rank K coefficients.
        key_states = attn.k_proj.u_proj(z_key_states)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        key_states = _apply_rope_to_k_standalone(
            key_states,
            cos,
            sin,
            position_ids,
        )

        key_for_scores = _repeat_kv(key_states, num_kv_groups)
        attn_scores = torch.matmul(query_states, key_for_scores.transpose(2, 3))
        attn_scores = attn_scores / math.sqrt(head_dim)
    elif k_is_lowrank:
        # Decode path: compute scores from compressed K without reconstruction.
        attn_scores = _lowrank_k_scores_from_compressed_cache(
            q_rope=query_states,
            z_key_cache=key_cache_out,
            u_for_heads=k_u_for_heads,
            cos=cos,
            sin=sin,
            head_dim=head_dim,
            chunk_size=k_score_chunk_size,
        )
    else:
        key_for_scores = _repeat_kv(key_cache_out, num_kv_groups)
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

    if v_is_lowrank:
        z_context = torch.einsum("bhqs,bsr->bhqr", attn_probs, value_cache_out)

        v_u_weight = attn.v_proj.u_proj.weight
        rank_v = int(attn.v_proj.rank)
        expected_v_out = num_kv_heads * head_dim

        if v_u_weight.shape[0] != expected_v_out:
            raise RuntimeError(
                f"Unexpected v_proj.u_proj output dimension: got {v_u_weight.shape[0]}, "
                f"expected {expected_v_out} = num_kv_heads({num_kv_heads}) * head_dim({head_dim})."
            )

        v_u_blocks = v_u_weight.view(num_kv_heads, head_dim, rank_v)
        v_u_for_heads = v_u_blocks[kv_head_for_query_head]

        attn_output = torch.einsum("bhqr,hdr->bhqd", z_context, v_u_for_heads)

        if attn.v_proj.u_proj.bias is not None:
            bias_blocks = attn.v_proj.u_proj.bias.view(num_kv_heads, head_dim)
            bias_for_heads = bias_blocks[kv_head_for_query_head]
            attn_output = attn_output + bias_for_heads.view(1, num_heads, 1, head_dim)
    else:
        value_for_context = _repeat_kv(value_cache_out, num_kv_groups)
        attn_output = torch.matmul(attn_probs, value_for_context)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    attn_output = attn.o_proj(attn_output)

    return attn_output, key_cache_out, value_cache_out


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
        past_len = _cache_seq_len(past_layer_caches[0][0])

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


def _allocate_lowrank_v_cache(model, batch_size: int, max_seq_len: int, device, dtype):
    """
    Preallocates per-layer K and V caches for the V-only path.

    Returns a list of dicts, one per decoder layer:
        {
            "v_lowrank": bool,
            "k":   [batch, num_kv_heads, max_seq_len, head_dim],
            "v_z": [batch, max_seq_len, rank_v]                   (if v_lowrank)
            "v":   [batch, num_kv_heads, max_seq_len, head_dim]   (if not v_lowrank)
        }
    """
    caches = []
    for layer in model.model.layers:
        attn = layer.self_attn
        num_heads, num_kv_heads, num_kv_groups, head_dim = _get_llama_attn_dims(attn)
        v_lowrank = _is_lowrank_module(attn.v_proj)

        k_buf = torch.empty(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )
        if v_lowrank:
            rank_v = int(attn.v_proj.rank)
            v_z_buf = torch.empty(
                batch_size, max_seq_len, rank_v,
                device=device, dtype=dtype,
            )
            v_buf = None
        else:
            v_z_buf = None
            v_buf = torch.empty(
                batch_size, num_kv_heads, max_seq_len, head_dim,
                device=device, dtype=dtype,
            )

        caches.append({
            "v_lowrank": v_lowrank,
            "k": k_buf,
            "v_z": v_z_buf,
            "v": v_buf,
        })

    return caches


def _lowrank_v_attention_forward_inplace(
    attn,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask_full: torch.Tensor,
    layer_cache: dict,
    write_pos: int,
    kv_seq_len: int,
    rotary_emb=None,
):
    """
    V-only attention with preallocated cache. Writes new K and V (or V_z)
    into layer_cache at slots [write_pos : write_pos + q_len] and reads the
    prefix [: kv_seq_len] for scoring.
    """
    bsz, q_len, _ = hidden_states.shape
    device = hidden_states.device
    num_heads, num_kv_heads, num_kv_groups, head_dim = _get_llama_attn_dims(attn)

    query_states = attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    rope = rotary_emb if rotary_emb is not None else getattr(attn, "rotary_emb", None)
    cos, sin = _get_rope_cos_sin(
        rotary_emb=rope,
        query_states=query_states,
        position_ids=position_ids,
        kv_seq_len=kv_seq_len,
    )
    query_states, key_states = _apply_rope_standalone(
        query_states, key_states, cos, sin, position_ids,
    )

    # In-place writes into preallocated cache.
    layer_cache["k"][:, :, write_pos:write_pos + q_len, :] = key_states

    v_lowrank = layer_cache["v_lowrank"]
    if v_lowrank:
        z_value_states = attn.v_proj.v_proj(hidden_states)
        layer_cache["v_z"][:, write_pos:write_pos + q_len, :] = z_value_states
    else:
        value_states = attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        layer_cache["v"][:, :, write_pos:write_pos + q_len, :] = value_states

    key_cache_prefix = layer_cache["k"][:, :, :kv_seq_len, :]
    key_for_scores = _repeat_kv(key_cache_prefix, num_kv_groups)

    attn_scores = torch.matmul(query_states, key_for_scores.transpose(2, 3))
    attn_scores = attn_scores / math.sqrt(head_dim)

    key_positions = torch.arange(kv_seq_len, device=device).view(1, 1, 1, kv_seq_len)
    query_positions = position_ids.view(bsz, 1, q_len, 1)
    causal_mask = key_positions <= query_positions
    attn_scores = attn_scores.masked_fill(~causal_mask, torch.finfo(attn_scores.dtype).min)

    if attention_mask_full is not None:
        key_padding_mask = attention_mask_full[:, None, None, :kv_seq_len].to(torch.bool)
        attn_scores = attn_scores.masked_fill(
            ~key_padding_mask,
            torch.finfo(attn_scores.dtype).min,
        )

    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)

    if v_lowrank:
        z_value_prefix = layer_cache["v_z"][:, :kv_seq_len, :]
        z_context = torch.einsum("bhqs,bsr->bhqr", attn_probs, z_value_prefix)

        u_weight = attn.v_proj.u_proj.weight
        rank_v = int(attn.v_proj.rank)
        u_blocks = u_weight.view(num_kv_heads, head_dim, rank_v)
        kv_head_for_query_head = torch.arange(num_heads, device=device) // num_kv_groups
        u_for_heads = u_blocks[kv_head_for_query_head]
        attn_output = torch.einsum("bhqr,hdr->bhqd", z_context, u_for_heads)

        if attn.v_proj.u_proj.bias is not None:
            bias_blocks = attn.v_proj.u_proj.bias.view(num_kv_heads, head_dim)
            bias_for_heads = bias_blocks[kv_head_for_query_head]
            attn_output = attn_output + bias_for_heads.view(1, num_heads, 1, head_dim)
    else:
        v_prefix = layer_cache["v"][:, :, :kv_seq_len, :]
        v_for_context = _repeat_kv(v_prefix, num_kv_groups)
        attn_output = torch.matmul(attn_probs, v_for_context)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    attn_output = attn.o_proj(attn_output)
    return attn_output


@torch.inference_mode()
def lowrank_v_cache_forward_inplace(
    model,
    input_ids,
    attention_mask_full,
    layer_caches,
    write_pos: int,
    kv_seq_len: int,
):
    decoder = model.model
    bsz, q_len = input_ids.shape
    device = input_ids.device
    decoder_rotary_emb = getattr(decoder, "rotary_emb", None)

    position_ids = torch.arange(
        write_pos, kv_seq_len, device=device, dtype=torch.long,
    ).unsqueeze(0).expand(bsz, -1)

    hidden_states = decoder.embed_tokens(input_ids)

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        residual = hidden_states
        hidden_states_norm = decoder_layer.input_layernorm(hidden_states)

        attn_output = _lowrank_v_attention_forward_inplace(
            attn=decoder_layer.self_attn,
            hidden_states=hidden_states_norm,
            position_ids=position_ids,
            attention_mask_full=attention_mask_full,
            layer_cache=layer_caches[layer_idx],
            write_pos=write_pos,
            kv_seq_len=kv_seq_len,
            rotary_emb=decoder_rotary_emb,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = decoder.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


@torch.inference_mode()
def lowrank_v_cache_generate_prealloc(model, input_ids, attention_mask, max_new_tokens):
    """
    Greedy generation using the preallocated V-only cache path.

    Allocates K and V (or V_z) cache buffers of size [batch, ..., prompt_len + max_new_tokens, ...]
    once at the start, then writes new K/V slices in-place at each step instead
    of growing via torch.cat.
    """
    bsz, prompt_len = input_ids.shape
    device = input_ids.device
    max_seq = prompt_len + max_new_tokens

    sample_attn = model.model.layers[0].self_attn
    weight_for_dtype = (
        sample_attn.q_proj.weight
        if isinstance(sample_attn.q_proj, nn.Linear)
        else sample_attn.q_proj.u_proj.weight
    )
    cache_dtype = weight_for_dtype.dtype

    layer_caches = _allocate_lowrank_v_cache(model, bsz, max_seq, device, cache_dtype)

    attention_mask_full = torch.ones(bsz, max_seq, device=device, dtype=attention_mask.dtype)
    attention_mask_full[:, :prompt_len] = attention_mask

    generated = torch.empty(bsz, max_seq, device=device, dtype=input_ids.dtype)
    generated[:, :prompt_len] = input_ids

    # Prefill writes [0:prompt_len].
    logits = lowrank_v_cache_forward_inplace(
        model=model,
        input_ids=input_ids,
        attention_mask_full=attention_mask_full,
        layer_caches=layer_caches,
        write_pos=0,
        kv_seq_len=prompt_len,
    )

    cur_pos = prompt_len

    for gen_idx in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated[:, cur_pos:cur_pos + 1] = next_token

        if gen_idx + 1 >= max_new_tokens:
            break

        logits = lowrank_v_cache_forward_inplace(
            model=model,
            input_ids=next_token,
            attention_mask_full=attention_mask_full,
            layer_caches=layer_caches,
            write_pos=cur_pos,
            kv_seq_len=cur_pos + 1,
        )
        cur_pos += 1

    return generated[:, :prompt_len + max_new_tokens]


def _allocate_lowrank_kv_cache(model, batch_size: int, max_seq_len: int, device, dtype):
    """
    Preallocates per-layer K and V caches for the compressed K+V path.

    Per-layer dict:
        "k_lowrank": bool
        "v_lowrank": bool
        "k_z" / "k": compressed or full K buffer
        "v_z" / "v": compressed or full V buffer
    """
    caches = []

    for layer in model.model.layers:
        attn = layer.self_attn
        num_heads, num_kv_heads, num_kv_groups, head_dim = _get_llama_attn_dims(attn)

        k_lowrank = _is_lowrank_module(attn.k_proj)
        v_lowrank = _is_lowrank_module(attn.v_proj)

        if k_lowrank:
            rank_k = int(attn.k_proj.rank)
            k_z_buf = torch.empty(batch_size, max_seq_len, rank_k, device=device, dtype=dtype)
            k_buf = None
        else:
            k_z_buf = None
            k_buf = torch.empty(
                batch_size, num_kv_heads, max_seq_len, head_dim,
                device=device, dtype=dtype,
            )

        if v_lowrank:
            rank_v = int(attn.v_proj.rank)
            v_z_buf = torch.empty(batch_size, max_seq_len, rank_v, device=device, dtype=dtype)
            v_buf = None
        else:
            v_z_buf = None
            v_buf = torch.empty(
                batch_size, num_kv_heads, max_seq_len, head_dim,
                device=device, dtype=dtype,
            )

        caches.append({
            "k_lowrank": k_lowrank,
            "v_lowrank": v_lowrank,
            "k_z": k_z_buf,
            "k": k_buf,
            "v_z": v_z_buf,
            "v": v_buf,
        })

    return caches


def _lowrank_kv_attention_forward_inplace(
    attn,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask_full: torch.Tensor,
    layer_cache: dict,
    write_pos: int,
    kv_seq_len: int,
    rotary_emb=None,
    k_score_chunk_size: int = 256,
):
    """
    Compressed K+V attention with preallocated cache. Writes new K/V or
    z_K/z_V into layer_cache at [write_pos : write_pos + q_len] and reads
    the prefix [: kv_seq_len].
    """
    bsz, q_len, _ = hidden_states.shape
    device = hidden_states.device

    num_heads, num_kv_heads, num_kv_groups, head_dim = _get_llama_attn_dims(attn)

    k_is_lowrank = layer_cache["k_lowrank"]
    v_is_lowrank = layer_cache["v_lowrank"]

    query_states = attn.q_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    rope = rotary_emb if rotary_emb is not None else getattr(attn, "rotary_emb", None)
    cos, sin = _get_rope_cos_sin(
        rotary_emb=rope,
        query_states=query_states,
        position_ids=position_ids,
        kv_seq_len=kv_seq_len,
    )

    # K write path
    if k_is_lowrank:
        query_states = _apply_rope_to_q_standalone(query_states, cos, sin, position_ids)

        z_key_states = attn.k_proj.v_proj(hidden_states)
        layer_cache["k_z"][:, write_pos:write_pos + q_len, :] = z_key_states

        k_u_weight = attn.k_proj.u_proj.weight
        rank_k = int(attn.k_proj.rank)
        expected_k_out = num_kv_heads * head_dim

        if k_u_weight.shape[0] != expected_k_out:
            raise RuntimeError(
                f"Unexpected k_proj.u_proj output dimension: got {k_u_weight.shape[0]}, "
                f"expected {expected_k_out} = num_kv_heads({num_kv_heads}) * head_dim({head_dim})."
            )

        k_u_blocks = k_u_weight.view(num_kv_heads, head_dim, rank_k)
        kv_head_for_query_head = torch.arange(num_heads, device=device) // num_kv_groups
        k_u_for_heads = k_u_blocks[kv_head_for_query_head]
    else:
        key_states = attn.k_proj(hidden_states)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        query_states, key_states = _apply_rope_standalone(
            query_states, key_states, cos, sin, position_ids,
        )

        layer_cache["k"][:, :, write_pos:write_pos + q_len, :] = key_states

    # V write path
    if v_is_lowrank:
        z_value_states = attn.v_proj.v_proj(hidden_states)
        layer_cache["v_z"][:, write_pos:write_pos + q_len, :] = z_value_states
    else:
        value_states = attn.v_proj(hidden_states)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        layer_cache["v"][:, :, write_pos:write_pos + q_len, :] = value_states

    kv_head_for_query_head = torch.arange(num_heads, device=device) // num_kv_groups

    # Score path
    if k_is_lowrank and write_pos == 0 and q_len == kv_seq_len:
        # Prefill: reconstruct only current prompt K, persistent cache stays compressed.
        key_states = attn.k_proj.u_proj(z_key_states)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        key_states = _apply_rope_to_k_standalone(key_states, cos, sin, position_ids)

        key_for_scores = _repeat_kv(key_states, num_kv_groups)
        attn_scores = torch.matmul(query_states, key_for_scores.transpose(2, 3))
        attn_scores = attn_scores / math.sqrt(head_dim)
    elif k_is_lowrank:
        # Decode: score directly from compressed K prefix.
        z_key_prefix = layer_cache["k_z"][:, :kv_seq_len, :]

        attn_scores = _lowrank_k_scores_from_compressed_cache(
            q_rope=query_states,
            z_key_cache=z_key_prefix,
            u_for_heads=k_u_for_heads,
            cos=cos,
            sin=sin,
            head_dim=head_dim,
            chunk_size=k_score_chunk_size,
        )
    else:
        key_prefix = layer_cache["k"][:, :, :kv_seq_len, :]
        key_for_scores = _repeat_kv(key_prefix, num_kv_groups)

        attn_scores = torch.matmul(query_states, key_for_scores.transpose(2, 3))
        attn_scores = attn_scores / math.sqrt(head_dim)

    key_positions = torch.arange(kv_seq_len, device=device).view(1, 1, 1, kv_seq_len)
    query_positions = position_ids.view(bsz, 1, q_len, 1)
    causal_mask = key_positions <= query_positions
    attn_scores = attn_scores.masked_fill(~causal_mask, torch.finfo(attn_scores.dtype).min)

    if attention_mask_full is not None:
        key_padding_mask = attention_mask_full[:, None, None, :kv_seq_len].to(torch.bool)
        attn_scores = attn_scores.masked_fill(
            ~key_padding_mask,
            torch.finfo(attn_scores.dtype).min,
        )

    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)

    # Value aggregation
    if v_is_lowrank:
        z_value_prefix = layer_cache["v_z"][:, :kv_seq_len, :]
        z_context = torch.einsum("bhqs,bsr->bhqr", attn_probs, z_value_prefix)

        v_u_weight = attn.v_proj.u_proj.weight
        rank_v = int(attn.v_proj.rank)
        expected_v_out = num_kv_heads * head_dim

        if v_u_weight.shape[0] != expected_v_out:
            raise RuntimeError(
                f"Unexpected v_proj.u_proj output dimension: got {v_u_weight.shape[0]}, "
                f"expected {expected_v_out} = num_kv_heads({num_kv_heads}) * head_dim({head_dim})."
            )

        v_u_blocks = v_u_weight.view(num_kv_heads, head_dim, rank_v)
        v_u_for_heads = v_u_blocks[kv_head_for_query_head]

        attn_output = torch.einsum("bhqr,hdr->bhqd", z_context, v_u_for_heads)

        if attn.v_proj.u_proj.bias is not None:
            bias_blocks = attn.v_proj.u_proj.bias.view(num_kv_heads, head_dim)
            bias_for_heads = bias_blocks[kv_head_for_query_head]
            attn_output = attn_output + bias_for_heads.view(1, num_heads, 1, head_dim)
    else:
        value_prefix = layer_cache["v"][:, :, :kv_seq_len, :]
        value_for_context = _repeat_kv(value_prefix, num_kv_groups)
        attn_output = torch.matmul(attn_probs, value_for_context)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    attn_output = attn.o_proj(attn_output)

    return attn_output


@torch.inference_mode()
def lowrank_kv_cache_forward_inplace(
    model,
    input_ids,
    attention_mask_full,
    layer_caches,
    write_pos: int,
    kv_seq_len: int,
    k_score_chunk_size: int = 256,
):
    decoder = model.model
    bsz, q_len = input_ids.shape
    device = input_ids.device
    decoder_rotary_emb = getattr(decoder, "rotary_emb", None)

    position_ids = torch.arange(
        write_pos, kv_seq_len, device=device, dtype=torch.long,
    ).unsqueeze(0).expand(bsz, -1)

    hidden_states = decoder.embed_tokens(input_ids)

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        residual = hidden_states
        hidden_states_norm = decoder_layer.input_layernorm(hidden_states)

        attn_output = _lowrank_kv_attention_forward_inplace(
            attn=decoder_layer.self_attn,
            hidden_states=hidden_states_norm,
            position_ids=position_ids,
            attention_mask_full=attention_mask_full,
            layer_cache=layer_caches[layer_idx],
            write_pos=write_pos,
            kv_seq_len=kv_seq_len,
            rotary_emb=decoder_rotary_emb,
            k_score_chunk_size=k_score_chunk_size,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = decoder.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


@torch.inference_mode()
def lowrank_kv_cache_generate_prealloc(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    k_score_chunk_size: int = 256,
):
    """
    Greedy generation using preallocated compressed K+V cache.
    """
    bsz, prompt_len = input_ids.shape
    device = input_ids.device
    max_seq = prompt_len + max_new_tokens

    sample_attn = model.model.layers[0].self_attn
    weight_for_dtype = (
        sample_attn.q_proj.weight
        if isinstance(sample_attn.q_proj, nn.Linear)
        else sample_attn.q_proj.u_proj.weight
    )
    cache_dtype = weight_for_dtype.dtype

    layer_caches = _allocate_lowrank_kv_cache(
        model=model,
        batch_size=bsz,
        max_seq_len=max_seq,
        device=device,
        dtype=cache_dtype,
    )

    attention_mask_full = torch.ones(bsz, max_seq, device=device, dtype=attention_mask.dtype)
    attention_mask_full[:, :prompt_len] = attention_mask

    generated = torch.empty(bsz, max_seq, device=device, dtype=input_ids.dtype)
    generated[:, :prompt_len] = input_ids

    logits = lowrank_kv_cache_forward_inplace(
        model=model,
        input_ids=input_ids,
        attention_mask_full=attention_mask_full,
        layer_caches=layer_caches,
        write_pos=0,
        kv_seq_len=prompt_len,
        k_score_chunk_size=k_score_chunk_size,
    )

    cur_pos = prompt_len

    for gen_idx in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated[:, cur_pos:cur_pos + 1] = next_token

        if gen_idx + 1 >= max_new_tokens:
            break

        logits = lowrank_kv_cache_forward_inplace(
            model=model,
            input_ids=next_token,
            attention_mask_full=attention_mask_full,
            layer_caches=layer_caches,
            write_pos=cur_pos,
            kv_seq_len=cur_pos + 1,
            k_score_chunk_size=k_score_chunk_size,
        )
        cur_pos += 1

    return generated[:, :prompt_len + max_new_tokens]


@torch.inference_mode()
def lowrank_kv_cache_forward(
    model,
    input_ids,
    attention_mask,
    past_layer_caches=None,
    k_score_chunk_size: int = 256,
):
    """
    Minimal standalone LLaMA forward pass using compressed K and compressed V cache.

    past_layer_caches:
        None, or list of length num_layers. Each entry is:
            (z_key_cache, z_value_cache)
    """
    decoder = model.model
    bsz, q_len = input_ids.shape
    device = input_ids.device

    decoder_rotary_emb = getattr(decoder, "rotary_emb", None)

    past_len = 0
    if past_layer_caches is not None and past_layer_caches[0] is not None:
        past_len = _cache_seq_len(past_layer_caches[0][0])

    position_ids = torch.arange(
        past_len,
        past_len + q_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).expand(bsz, -1)

    hidden_states = decoder.embed_tokens(input_ids)
    new_layer_caches = []

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        past_z_key = None
        past_z_value = None

        if past_layer_caches is not None:
            past_z_key, past_z_value = past_layer_caches[layer_idx]

        residual = hidden_states
        hidden_states_norm = decoder_layer.input_layernorm(hidden_states)

        attn_output, new_z_key, new_z_value = _lowrank_kv_attention_forward(
            attn=decoder_layer.self_attn,
            hidden_states=hidden_states_norm,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_z_key=past_z_key,
            past_z_value=past_z_value,
            rotary_emb=decoder_rotary_emb,
            k_score_chunk_size=k_score_chunk_size,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        new_layer_caches.append((new_z_key, new_z_value))

    hidden_states = decoder.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return logits, new_layer_caches


@torch.inference_mode()
def lowrank_kv_cache_generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    k_score_chunk_size: int = 256,
):
    """
    Greedy generation using compressed K and compressed V cache.
    """
    generated = input_ids

    logits, layer_caches = lowrank_kv_cache_forward(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_layer_caches=None,
        k_score_chunk_size=k_score_chunk_size,
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

        logits, layer_caches = lowrank_kv_cache_forward(
            model=model,
            input_ids=next_token,
            attention_mask=attention_mask,
            past_layer_caches=layer_caches,
            k_score_chunk_size=k_score_chunk_size,
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
    cache_batch_size = None
    cache_prompt_len = None

    for step, (input_ids,) in enumerate(loader):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = torch.ones_like(input_ids, device=device)

        if cache_batch_size is None:
            cache_batch_size = int(input_ids.shape[0])
            cache_prompt_len = int(input_ids.shape[1])

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

    summary = build_memory_breakdown_summary(
        model=model,
        total_time=total_time,
        total_new_tokens=total_new_tokens,
        weight_mem_bytes=weight_mem,
        peak_mem_bytes=peak_mem,
        cache_batch_size=cache_batch_size,
        cache_seq_len=cache_prompt_len + gen_len,
        cache_kind="hf_full",
    )

    return summary, rows


@torch.inference_mode()
def benchmark_generate_lowrank_v_cache(
    model, tokenizer, loader, device, gen_len, num_batches, warmup,
    prealloc: bool = False,
):
    if device.type != "cuda":
        raise RuntimeError("This benchmark script expects CUDA.")

    _validate_lowrank_v_cache_model(model)

    if prealloc:
        print("[lowrank-v-cache] using preallocated cache path")
        gen_fn = lowrank_v_cache_generate_prealloc
    else:
        gen_fn = lowrank_v_cache_generate

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    weight_mem = torch.cuda.memory_allocated(device)

    total_time = 0.0
    total_new_tokens = 0
    rows = []
    cache_batch_size = None
    cache_prompt_len = None

    for step, (input_ids,) in enumerate(loader):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = torch.ones_like(input_ids, device=device)

        if cache_batch_size is None:
            cache_batch_size = int(input_ids.shape[0])
            cache_prompt_len = int(input_ids.shape[1])

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        _ = gen_fn(
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

    summary = build_memory_breakdown_summary(
        model=model,
        total_time=total_time,
        total_new_tokens=total_new_tokens,
        weight_mem_bytes=weight_mem,
        peak_mem_bytes=peak_mem,
        cache_batch_size=cache_batch_size,
        cache_seq_len=cache_prompt_len + gen_len,
        cache_kind="lowrank_v",
    )

    return summary, rows


@torch.inference_mode()
def benchmark_generate_lowrank_kv_cache(
    model,
    tokenizer,
    loader,
    device,
    gen_len,
    num_batches,
    warmup,
    k_score_chunk_size: int = 256,
    prealloc: bool = False,
):
    if device.type != "cuda":
        raise RuntimeError("This benchmark script expects CUDA.")

    _validate_lowrank_kv_cache_model(model)

    if prealloc:
        print("[lowrank-kv-cache] using preallocated cache path")
        gen_fn = lowrank_kv_cache_generate_prealloc
    else:
        gen_fn = lowrank_kv_cache_generate

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    weight_mem = torch.cuda.memory_allocated(device)

    total_time = 0.0
    total_new_tokens = 0
    rows = []
    cache_batch_size = None
    cache_prompt_len = None

    for step, (input_ids,) in enumerate(loader):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = torch.ones_like(input_ids, device=device)

        if cache_batch_size is None:
            cache_batch_size = int(input_ids.shape[0])
            cache_prompt_len = int(input_ids.shape[1])

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        _ = gen_fn(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_len,
            k_score_chunk_size=k_score_chunk_size,
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

    summary = build_memory_breakdown_summary(
        model=model,
        total_time=total_time,
        total_new_tokens=total_new_tokens,
        weight_mem_bytes=weight_mem,
        peak_mem_bytes=peak_mem,
        cache_batch_size=cache_batch_size,
        cache_seq_len=cache_prompt_len + gen_len,
        cache_kind="lowrank_kv",
    )

    return summary, rows


def slugify(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").strip("_") or "unknown"


_GPR_PATTERNS = [
    r"gpr(\d+p\d+)",
    r"gpr(\d+\.\d+)",
    r"prune_rate_(\d+\.\d+)",
    r"prune_rate_(\d+p\d+)",
    r"pr(\d+p\d+)",
    r"pr(\d+\.\d+)",
]


def infer_gpr_tag(*candidates: str) -> str:
    for s in candidates:
        if not s:
            continue
        for pat in _GPR_PATTERNS:
            m = re.search(pat, s)
            if m:
                return m.group(1).replace(".", "p")
    return ""


def build_auto_label(model_kind: str, gpr_tag: str, prompt_len: int, gen_len: int, batch_size: int) -> str:
    parts = [model_kind]
    if gpr_tag:
        parts.append(gpr_tag)
    parts.extend([f"p{prompt_len}", f"g{gen_len}", f"b{batch_size}"])
    return "_".join(parts)


def build_run_filename(label: str, cache_mode: str, timestamp: str) -> str:
    return f"{slugify(label)}_{cache_mode}_{timestamp}.json"


def write_run_json(out_dir: str, filename: str, payload: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


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
    ap.add_argument(
        "--model_kind",
        choices=["dense", "lowrank", "auto"],
        default="auto",
        help=(
            "auto (default): pick dense if --pt_path is empty, else lowrank. "
            "dense: load --dense_model from HuggingFace. "
            "lowrank: export from --pt_path then load."
        ),
    )

    # Dense path
    ap.add_argument("--dense_model", default="meta-llama/Llama-2-7b-hf")

    # Low-rank export inputs
    ap.add_argument("--pt_path", default="")
    ap.add_argument("--export_dir", default="")
    ap.add_argument("--tokenizer_name", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--force_export", action="store_true")
    ap.add_argument("--skip_export", action="store_true")

    # Benchmark metadata
    ap.add_argument("--label", default="", help="Optional label override. If empty, auto-derived as e.g. lowrank_0p6_p1024_g1024_b1.")
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
    ap.add_argument(
        "--vcache_prealloc",
        action="store_true",
        help=(
            "Use preallocated K/V cache buffers for the V-only path "
            "(avoids torch.cat per decode step). Combine with --lowrank_v_cache."
        ),
    )
    ap.add_argument(
        "--kvcache_prealloc",
        action="store_true",
        help=(
            "Use preallocated K/V cache buffers for the compressed K+V path "
            "(avoids torch.cat per decode step). Combine with --lowrank_k_cache."
        ),
    )
    ap.add_argument(
        "--lowrank_k_cache",
        action="store_true",
        help=(
            "Use standalone compressed K+V cache generation. "
            "Requires --model_kind lowrank and LowRankLinear self_attn.k_proj/self_attn.v_proj modules."
        ),
    )
    ap.add_argument(
        "--k_score_chunk_size",
        type=int,
        default=256,
        help="Chunk size used when computing decode attention scores from compressed K cache.",
    )

    # Output
    ap.add_argument(
        "--out_dir",
        default="./results",
        help="Directory to write the per-run timestamped JSON file.",
    )

    args = ap.parse_args()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_command = " ".join(shlex.quote(a) for a in sys.argv)

    # Resolve auto model_kind: lowrank if a checkpoint is supplied, else dense.
    if args.model_kind == "auto":
        if args.pt_path or args.skip_export:
            args.model_kind = "lowrank"
            print(f"[model_kind] auto -> lowrank (pt_path={args.pt_path or '<skip_export>'})")
        else:
            args.model_kind = "dense"
            print(f"[model_kind] auto -> dense ({args.dense_model})")

    gpr_tag = infer_gpr_tag(args.pt_path, args.export_dir) if args.model_kind == "lowrank" else ""
    label = args.label or build_auto_label(
        model_kind=args.model_kind,
        gpr_tag=gpr_tag,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        batch_size=args.batch_size,
    )

    set_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if device.type != "cuda":
        raise RuntimeError("CUDA device required for this benchmark.")

    if args.lowrank_v_cache and args.model_kind != "lowrank":
        raise ValueError("--lowrank_v_cache requires --model_kind lowrank")

    if args.lowrank_k_cache and args.model_kind != "lowrank":
        raise ValueError("--lowrank_k_cache requires --model_kind lowrank")

    if args.vcache_prealloc and not args.lowrank_v_cache:
        raise ValueError("--vcache_prealloc requires --lowrank_v_cache")

    if args.kvcache_prealloc and not args.lowrank_k_cache:
        raise ValueError("--kvcache_prealloc requires --lowrank_k_cache")

    if args.lowrank_k_cache and args.lowrank_v_cache:
        raise ValueError("Use either --lowrank_v_cache or --lowrank_k_cache, not both.")

    if (args.lowrank_v_cache or args.lowrank_k_cache) and args.compile:
        raise ValueError("Standalone low-rank cache paths currently do not support --compile")

    free, total = torch.cuda.mem_get_info(device)
    print(f"[preflight] CUDA free={free / 2**30:.2f} GB / total={total / 2**30:.2f} GB")

    print("=" * 80)
    print("label:", label)
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
    print("vcache_prealloc:", args.vcache_prealloc)
    print("kvcache_prealloc:", args.kvcache_prealloc)
    print("lowrank_k_cache:", args.lowrank_k_cache)
    print("k_score_chunk_size:", args.k_score_chunk_size)
    print("batch_size:", args.batch_size)
    print("=" * 80)

    # 1. Export if low-rank
    if args.model_kind == "lowrank":
        if not args.export_dir:
            raise ValueError("--export_dir is required when --model_kind lowrank")

        if not args.skip_export:
            if not args.pt_path:
                raise ValueError("--pt_path is required for lowrank export unless --skip_export is passed")

            if os.path.isfile(args.pt_path):
                size_gb = os.path.getsize(args.pt_path) / 2**30
                print(f"[checkpoint] compressed.pt model FOUND")
                print(f"[checkpoint] path: {os.path.abspath(args.pt_path)}")
                print(f"[checkpoint] size: {size_gb:.2f} GB")
            else:
                raise FileNotFoundError(
                    f"--pt_path does not exist: {args.pt_path}"
                )

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
        model_path = args.dense_model
        model_id = os.path.basename(args.dense_model.rstrip("/"))
    else:
        model, tokenizer = load_lowrank_model(args.export_dir, dtype=dtype, device=device)
        model_path = args.export_dir
        model_id = os.path.basename(args.export_dir.rstrip("/"))

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
    if args.lowrank_k_cache:
        summary, batch_rows = benchmark_generate_lowrank_kv_cache(
            model=model,
            tokenizer=tokenizer,
            loader=loader,
            device=device,
            gen_len=args.gen_len,
            num_batches=args.num_batches,
            warmup=args.warmup,
            k_score_chunk_size=args.k_score_chunk_size,
            prealloc=args.kvcache_prealloc,
        )
        kv_cache_impl = (
            "lowrank_kv_cache_phase2_prealloc"
            if args.kvcache_prealloc
            else "lowrank_kv_cache_phase2"
        )
    elif args.lowrank_v_cache:
        summary, batch_rows = benchmark_generate_lowrank_v_cache(
            model=model,
            tokenizer=tokenizer,
            loader=loader,
            device=device,
            gen_len=args.gen_len,
            num_batches=args.num_batches,
            warmup=args.warmup,
            prealloc=args.vcache_prealloc,
        )
        kv_cache_impl = (
            "lowrank_v_cache_phase1_prealloc"
            if args.vcache_prealloc
            else "lowrank_v_cache_phase1"
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
        kv_cache_impl = "hf_generate_default"

    if args.lowrank_k_cache:
        cache_mode_tag = "kv_both_prealloc" if args.kvcache_prealloc else "kv_both"
    elif args.lowrank_v_cache:
        cache_mode_tag = "v_only_prealloc" if args.vcache_prealloc else "v_only"
    else:
        cache_mode_tag = "vanilla"

    result_row = {
        "label": label,
        "model_id": model_id,
        "gpr": gpr_tag,
        "cache_mode": cache_mode_tag,
        "kv_cache_impl": kv_cache_impl,
        "model_kind": args.model_kind,
        "model_path": model_path,
        "pt_path": args.pt_path,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "compile": args.compile,
        "k_score_chunk_size": args.k_score_chunk_size if args.lowrank_k_cache else "",
        "vcache_prealloc": args.vcache_prealloc,
        "kvcache_prealloc": args.kvcache_prealloc,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "warmup": args.warmup,
        **summary,
    }

    memory_breakdown = {
        "weight_mem_gb": result_row["weight_mem_gb"],
        "kv_cache_mem_gb": result_row["kv_cache_mem_gb"],
        "other_runtime_mem_gb": result_row["other_runtime_mem_gb"],
        "non_weight_runtime_mem_gb": result_row["non_weight_runtime_mem_gb"],
        "peak_mem_gb": result_row["peak_mem_gb"],
        "weight_mem_bytes": result_row["weight_mem_bytes"],
        "kv_cache_mem_bytes": result_row["kv_cache_mem_bytes"],
        "other_runtime_mem_bytes": result_row["other_runtime_mem_bytes"],
        "non_weight_runtime_mem_bytes": result_row["non_weight_runtime_mem_bytes"],
        "memory_cache_kind": result_row["memory_cache_kind"],
    }

    payload = {
        "timestamp": run_timestamp,
        "command": run_command,
        "argv": sys.argv,
        "args": vars(args),
        "summary": result_row,
        "memory_breakdown": memory_breakdown,
        "per_batch": batch_rows,
    }

    filename = build_run_filename(
        label=label,
        cache_mode=cache_mode_tag,
        timestamp=run_timestamp,
    )
    out_path = write_run_json(args.out_dir, filename, payload)

    print("\nSUMMARY")
    for k, v in result_row.items():
        print(f"{k}: {v}")

    print("\nwrote:")
    print(out_path)


if __name__ == "__main__":
    main()