"""
Usage:
    python evaluate_speed_standalone.py --model_path <path_to_model>
"""

import argparse
import os
import json
import logging
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random
import itertools
import time
from tqdm import tqdm

# Custom Modules (extracted from Dobi-SVD/modules/)
class SVDTransformLayer(nn.Module):
    """SVD-based low-rank layer decomposition."""
    def __init__(self, gamma=None, weight=None, bias=None, name=None, device=None):
        super(SVDTransformLayer, self).__init__()
        
        if name:
            self.name = name
        
        W_T = weight.T.to(torch.float32)
        U, S, V = torch.svd_lowrank(W_T, q=int(gamma), niter=10)
        diag_S = torch.diag(S)
        sqrt_S = torch.sqrt(diag_S)
        A_weight_T = (U @ sqrt_S).to(torch.float16)
        B_weight_T = (sqrt_S @ V.T).to(torch.float16)
        
        if bias is None: 
            self.ALinear = nn.Linear(W_T.size(0), gamma, bias=False, device=device)
            self.BLinear = nn.Linear(gamma, W_T.size(1), bias=False, device=device)
        else:
            self.ALinear = nn.Linear(W_T.size(0), gamma, bias=False, device=device)
            self.BLinear = nn.Linear(gamma, W_T.size(1), bias=True, device=device)
            self.BLinear.bias = nn.Parameter(bias).to(device)
       
        self.ALinear.weight = nn.Parameter(A_weight_T.T.contiguous()).to(device) 
        self.BLinear.weight = nn.Parameter(B_weight_T.T.contiguous()).to(device) 
                
    def forward(self, x):
        x = self.ALinear(x)
        x = self.BLinear(x)
        return x


class SVDTransformLayer_remapping(nn.Module):
    """SVD layer with remapping support."""
    def __init__(self, weight1=None, weight2=None, bias=None, name=None, device=None):
        super(SVDTransformLayer_remapping, self).__init__()
        
        if name:
            self.name = name
        
        A_weight_T = weight1.to(torch.float16)
        B_weight_T = weight2.to(torch.float16)
        
        if bias is None: 
            self.ALinear = nn.Linear(weight1.size(0), weight1.size(1), bias=False, device=device)
            self.BLinear = nn.Linear(weight2.size(0), weight2.size(1), bias=False, device=device)
        else:
            self.ALinear = nn.Linear(weight1.size(0), weight1.size(1), bias=False, device=device)
            self.BLinear = nn.Linear(weight2.size(0), weight2.size(1), bias=True, device=device)
            self.BLinear.bias = nn.Parameter(bias).to(device)
       
        self.ALinear.weight = nn.Parameter(A_weight_T.T.contiguous()).to(device) 
        self.BLinear.weight = nn.Parameter(B_weight_T.T.contiguous()).to(device) 
                
    def forward(self, x):
        x = self.ALinear(x)
        x = self.BLinear(x)
        return x


def DOBI_dequantize(us_quan, vt_quan, us_absmax, vt_absmax, tuple_info, code=None):
    """Dequantize DOBI compressed matrices."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("bitsandbytes is required for remapping models. Install it with: pip install bitsandbytes")
    
    if code is None:
        code = bnb.functional.create_dynamic_map().to("cuda")
    
    # Dequantize
    us_absmax = us_absmax.to(torch.float32).to("cuda")
    vt_absmax = vt_absmax.to(torch.float32).to("cuda")
    us_quan = us_quan.to("cuda")
    vt_quan = vt_quan.to("cuda")
    
    dequan_us = bnb.functional.dequantize_no_absmax(us_quan, code=code)
    dequan_us = dequan_us.T * us_absmax
    dequan_vt = bnb.functional.dequantize_no_absmax(vt_quan, code=code)
    dequan_vt = torch.diag(vt_absmax) @ dequan_vt

    if tuple_info is not None:
        if tuple_info[0] == "us":
            dequan_us = torch.cat((dequan_us, tuple_info[1].to("cuda")), dim=0)
        elif tuple_info[0] == "vt":
            dequan_vt = torch.cat((dequan_vt, tuple_info[1].to("cuda")), dim=1)
    
    dequan_us = dequan_us.to(torch.float16)  
    dequan_vt = dequan_vt.to(torch.float16) 
    return dequan_us, dequan_vt

import sys, types
import torch
import torch.nn as nn

# ---- TOP-LEVEL class so it is picklable ----
class LowRankLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # minimal attributes to satisfy unpickling
        self.weight = nn.Parameter(torch.empty(0))
        self.bias = None

    def forward(self, x):
        raise RuntimeError("LowRankLinear shim should never be executed.")

def _install_unpickle_shims():
    if "my_utils" not in sys.modules:
        sys.modules["my_utils"] = types.ModuleType("my_utils")
    m = sys.modules["my_utils"]
    m.LowRankLinear = LowRankLinear
_install_unpickle_shims()

'''
Model loading
'''
def load_remapping_model(updated_model_path):
    """Load a model with remapping/quantization."""
    logging.getLogger("transformers").setLevel(logging.ERROR)
        
    model_id = updated_model_path
    config = AutoConfig.from_pretrained(f"{model_id}/config.json")
    model = AutoModelForCausalLM.from_config(config)
    state_dict = torch.load(f"{model_id}/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.float16) 
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    mapping_info = torch.load(f"{model_id}/remapping_weight.pt", map_location="cpu")
    
    for name, module in tqdm(model.named_modules(), desc="Dequantize the model after remapping"):
        if isinstance(module, nn.Linear) and all(x not in name for x in ['lm_head']):
            us_quan = mapping_info[name]["us_quan"]
            vt_quan = mapping_info[name]["vt_quan"]
            us_absmax = mapping_info[name]["us_absmax"]
            vt_absmax = mapping_info[name]["vt_absmax"]
            tuple_info = mapping_info[name]["tuple_info"]
            dequan_us, dequan_vt = DOBI_dequantize(us_quan, vt_quan, us_absmax, vt_absmax, tuple_info, code=None)

            compress_size = dequan_vt.size(0) * dequan_vt.size(1) + dequan_us.size(0) * dequan_us.size(1)
            ori_size = module.in_features * module.out_features
            
            if ori_size > compress_size:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                NewLayer = SVDTransformLayer_remapping(
                    weight1=dequan_vt.T, 
                    weight2=dequan_us.T,
                    bias=module.bias, 
                    name=name, 
                    device="cuda"
                )
                setattr(parent, attr_name, NewLayer)
                del module
            else:
                new_weight = dequan_us @ dequan_vt
                module.weight.data = new_weight.detach()
                
            mapping_info[name] = {}
            
    return model, tokenizer 


def load_unremapping_model(model_id):
    """Load a model saved with DobiSVD."""
    pruned_dict = torch.load(f"{model_id}/DobiSVD_Model.pt", weights_only=False)
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

from collections import OrderedDict

def _strip_prefix_if_present(state, prefix):
    if isinstance(state, dict) and state and all(isinstance(k, str) and k.startswith(prefix) for k in state.keys()):
        return OrderedDict((k[len(prefix):], v) for k, v in state.items())
    return state

def _is_tensor_state_dict(obj):
    return isinstance(obj, dict) and obj and all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()
    )

def load_stage_fp16_checkpoint(model_dir, stage_pt="stage01_after_truncation_fp16.pt"):
    cfg_path = next(
        p for p in os.listdir(model_dir)
        if p.startswith("experiment_config") and p.endswith(".json")
    )
    with open(os.path.join(model_dir, cfg_path), "r") as f:
        cfg = json.load(f)
    base_model_id = cfg["model"]

    config = AutoConfig.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = os.path.join(model_dir, stage_pt)
    clean_path = os.path.join(model_dir, "stage01_state_dict_fp16.pt")

    # --- 1) Try to load an existing tensor-only state_dict safely ---
    state = None
    if os.path.exists(clean_path):
        try:
            maybe = torch.load(clean_path, map_location="cpu", weights_only=True)
            if isinstance(maybe, dict) and "state_dict" in maybe and isinstance(maybe["state_dict"], dict):
                maybe = maybe["state_dict"]
            if _is_tensor_state_dict(maybe):
                state = maybe
            else:
                raise RuntimeError("clean_path exists but is not a pure tensor state_dict")
        except Exception as e:
            print(f"[warn] {clean_path} not weights-only loadable: {e}")
            state = None

    # --- 2) If not available/valid, unpickle original checkpoint ONCE and extract state_dict ---
    if state is None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if hasattr(ckpt, "state_dict") and not isinstance(ckpt, dict):
            state = ckpt.state_dict()
        elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError(f"Unknown checkpoint type: {type(ckpt)}")

        # If somehow still model-like
        if hasattr(state, "state_dict") and not isinstance(state, dict):
            state = state.state_dict()

        if not isinstance(state, dict):
            raise ValueError(f"Extracted state is not a dict; got {type(state)}")

        # Save tensor-only state_dict for future safe loads
        torch.save(state, clean_path)
        print("Saved clean tensor-only state_dict to:", clean_path)

    # Normalize common key patterns
    state = _strip_prefix_if_present(state, "model.")
    state = _strip_prefix_if_present(state, "module.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_stage_fp16_checkpoint] missing={len(missing)} unexpected={len(unexpected)}")

    model = model.half()
    return model, tokenizer

def load_standard_model(model_path):
    """Load a standard HuggingFace model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# Data Loading
class IndexDataset(Dataset):
    """Simple dataset wrapper for tokenized sequences."""
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    """Load and prepare test dataset."""
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Evaluation Function
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, 
             batch_size=1, device="cuda"):
    """Evaluate model generation speed and memory usage."""
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)
    weight_memory = torch.cuda.memory_allocated()

    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()

        try:
            generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len + generated_len,
                top_p=0.95,
                temperature=1,
            )
            torch.cuda.synchronize()
            end_time = time.time()
            end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)

            if torch.isfinite(generation_output).all():
                throughput += (end_time - start_time)
                print(f"Batch {batch_idx+1}: Time {end_time - start_time:.2f} sec")

        except RuntimeError as e:
            print(f"Error during generation: {e}")
            torch.cuda.empty_cache()
            token_num -= batch.shape[0] * generated_len

    total_memory_gb = end_memory / (1024 ** 3)
    weight_memory_gb = weight_memory / (1024 ** 3)
    activation_memory_gb = (end_memory - start_memory) / (1024 ** 3)

    print(f"\n{'='*60}")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print(f"Weight Memory: {weight_memory_gb:.2f} GB")
    print(f"Activation Memory: {activation_memory_gb:.2f} GB")
    
    if throughput > 0:
        print(f"Throughput: {token_num / throughput:.2f} tokens/sec")
    else:
        print("Throughput could not be calculated due to errors.")
    print(f"{'='*60}\n")

# Main Function
def main(args):
    """Main evaluation function."""
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    DEV_GPU = torch.device('cuda:0')
    
    print(f"Loading model from: {args.model_path}")
    print(f"Model type: {'remapping' if args.remapping else 'standard/DobiSVD'}")
    
    # Load model based on type
    if args.remapping:
        model, tokenizer = load_remapping_model(args.model_path)
    elif os.path.exists(os.path.join(args.model_path, "DobiSVD_Model.pt")):
        print("Detected DobiSVD_Model.pt - loading with load_unremapping_model")
        model, tokenizer = load_unremapping_model(args.model_path)
    elif os.path.exists(os.path.join(args.model_path, "stage01_after_truncation_fp16.pt")):
        print("Detected stage checkpoint - loading with load_stage_fp16_checkpoint")
        model, tokenizer = load_stage_fp16_checkpoint(args.model_path, "stage01_after_truncation_fp16.pt")
    else:
        print("Loading as standard HuggingFace model")
        model, tokenizer = load_standard_model(args.model_path)

    # Move model to GPU
    model.to(DEV_GPU)
    print(f"Model loaded and moved to {DEV_GPU}")
    
    # Run evaluation
    eff_eval(
        model, 
        tokenizer, 
        dataset=args.eval_dataset, 
        generated_len=args.generated_len, 
        batch_size=args.batch_size, 
        device=DEV_GPU
    )

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone script to evaluate model generation speed and memory usage"
    )
    
    # Model settings
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (can be HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--remapping",
        action="store_true",
        default=False,
        help="Whether to use remapping to load model (requires bitsandbytes)",
    )
    
    # Device settings
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    
    # Dataset settings
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--generated_len",
        type=int,
        default=64,
        help="Length of generated tokens",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    
    args = parser.parse_args()
    main(args)
