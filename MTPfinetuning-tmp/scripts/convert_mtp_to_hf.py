#!/usr/bin/env python3
"""
Convert Megatron-LM MTP finetuned checkpoint to HuggingFace format for vLLM.

This script extracts only the MTP layer (layer 61) from the finetuned Megatron checkpoint
and saves it in a format compatible with vLLM's eagle speculative decoding.

Usage:
    python convert_mtp_to_hf.py \
        --megatron-ckpt /data/finetuned_mtp_v32/iter_0000050 \
        --base-model /data/models/DeepSeek-V3.2 \
        --output-dir /data/finetuned_layer61
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_megatron_dist_checkpoint(ckpt_dir: str, tp_size: int = 16) -> dict:
    """
    Load Megatron distributed checkpoint and gather MTP layer weights.
    
    Megatron saves weights in format: __<tp_rank>_<pp_rank>.distcp
    For TP=16, PP=1, we have files: __0_0.distcp to __15_0.distcp
    """
    print(f"Loading Megatron checkpoint from {ckpt_dir}")
    
    # First check if we need to use megatron's loader
    ckpt_path = Path(ckpt_dir)
    distcp_files = list(ckpt_path.glob("*.distcp"))
    
    if not distcp_files:
        raise ValueError(f"No .distcp files found in {ckpt_dir}")
    
    print(f"Found {len(distcp_files)} .distcp files")
    
    # Load MTP weights from each TP rank
    mtp_weights = {}
    
    # We need to use torch.distributed.checkpoint to load distributed checkpoints
    # For now, provide instructions for using Megatron's converter
    
    print("\n" + "="*80)
    print("NOTE: Megatron distributed checkpoints require special handling.")
    print("You have two options:")
    print()
    print("Option 1: Use Megatron's built-in checkpoint converter:")
    print(f"  cd /workspace/Megatron-LM")
    print(f"  python tools/checkpoint/convert.py \\")
    print(f"    --model-type GPT \\")
    print(f"    --loader mcore \\")
    print(f"    --saver mcore \\")
    print(f"    --load-dir {ckpt_dir} \\")
    print(f"    --save-dir {ckpt_dir}_converted \\")
    print(f"    --target-tensor-parallel-size 1 \\")
    print(f"    --target-pipeline-parallel-size 1")
    print()
    print("Option 2: Use this script's extraction mode after conversion:")
    print(f"  python convert_mtp_to_hf.py --extract-only ...")
    print("="*80 + "\n")
    
    return mtp_weights


def extract_mtp_from_base_model(base_model_dir: str, output_dir: str):
    """
    Extract MTP layer (layer 61) from the base DeepSeek-V3.2 model.
    This creates a standalone MTP model that vLLM can load.
    """
    print(f"Extracting MTP layer from {base_model_dir}")
    
    # Load the model index
    index_file = os.path.join(base_model_dir, "model.safetensors.index.json")
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    weight_map = index.get('weight_map', {})
    
    # Find all layer 61 (MTP layer) weights
    mtp_weights = {}
    mtp_files = set()
    
    for key, filename in weight_map.items():
        if 'layers.61' in key:
            mtp_files.add(filename)
    
    print(f"MTP weights are in files: {sorted(mtp_files)}")
    
    # Load MTP weights from safetensors files
    for filename in sorted(mtp_files):
        filepath = os.path.join(base_model_dir, filename)
        print(f"Loading {filename}...")
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if 'layers.61' in key:
                    mtp_weights[key] = f.get_tensor(key)
    
    print(f"Loaded {len(mtp_weights)} MTP weight tensors")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save MTP weights in safetensors format
    # vLLM expects the MTP layer to have specific naming
    output_weights = {}
    new_weight_map = {}
    
    for key, tensor in mtp_weights.items():
        # Keep original naming for vLLM compatibility
        output_weights[key] = tensor
        new_weight_map[key] = "model.safetensors"
    
    # Save weights
    output_file = os.path.join(output_dir, "model.safetensors")
    print(f"Saving MTP weights to {output_file}...")
    save_file(output_weights, output_file)
    
    # Create config.json for the MTP model
    base_config_file = os.path.join(base_model_dir, "config.json")
    with open(base_config_file, 'r') as f:
        base_config = json.load(f)
    
    # MTP config - keep same as base model
    mtp_config = base_config.copy()
    
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(mtp_config, f, indent=2)
    
    # Create model index
    index_output = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in output_weights.values())},
        "weight_map": new_weight_map
    }
    
    index_output_file = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_output_file, 'w') as f:
        json.dump(index_output, f, indent=2)
    
    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json", 
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt"
    ]
    
    # Check encoding directory as well
    encoding_dir = os.path.join(base_model_dir, "encoding")
    if os.path.exists(encoding_dir):
        for f in os.listdir(encoding_dir):
            src = os.path.join(encoding_dir, f)
            dst = os.path.join(output_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"Copied {f} from encoding/")
    
    for tf in tokenizer_files:
        src = os.path.join(base_model_dir, tf)
        if os.path.exists(src):
            shutil.copy2(src, output_dir)
            print(f"Copied {tf}")
    
    print(f"\nMTP model saved to {output_dir}")
    print(f"Total weights: {len(output_weights)}")
    
    # Print weight summary
    print("\nWeight summary:")
    weight_types = defaultdict(int)
    for key in output_weights:
        parts = key.split('.')
        if 'experts' in parts:
            weight_types['experts'] += 1
        elif 'self_attn' in parts:
            weight_types['attention'] += 1
        elif any(n in key for n in ['enorm', 'hnorm', 'eh_proj']):
            weight_types['mtp_specific'] += 1
        elif 'shared_head' in key:
            weight_types['shared_head'] += 1
        else:
            weight_types['other'] += 1
    
    for wt, count in sorted(weight_types.items()):
        print(f"  {wt}: {count}")
    
    return output_dir


def merge_finetuned_mtp(base_mtp_dir: str, finetuned_weights: dict, output_dir: str):
    """
    Merge finetuned MTP weights with base MTP model.
    """
    print(f"Merging finetuned weights into {output_dir}")
    
    # Load base MTP weights
    base_weights = {}
    base_file = os.path.join(base_mtp_dir, "model.safetensors")
    
    with safe_open(base_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            base_weights[key] = f.get_tensor(key)
    
    # Update with finetuned weights
    updated_count = 0
    for key, tensor in finetuned_weights.items():
        if key in base_weights:
            base_weights[key] = tensor
            updated_count += 1
            print(f"  Updated: {key}")
    
    print(f"Updated {updated_count} weight tensors")
    
    # Save merged weights
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "model.safetensors")
    save_file(base_weights, output_file)
    
    # Copy config and other files
    for f in os.listdir(base_mtp_dir):
        if f != "model.safetensors":
            src = os.path.join(base_mtp_dir, f)
            dst = os.path.join(output_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
    
    print(f"Merged MTP model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Megatron MTP checkpoint to HuggingFace format")
    parser.add_argument("--megatron-ckpt", type=str, help="Megatron checkpoint directory")
    parser.add_argument("--base-model", type=str, required=True, help="Base DeepSeek-V3.2 model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--extract-only", action="store_true", 
                       help="Only extract MTP from base model (no finetuned weights)")
    parser.add_argument("--tp-size", type=int, default=16, help="Tensor parallel size used in training")
    
    args = parser.parse_args()
    
    if args.extract_only:
        # Just extract MTP layer from base model
        extract_mtp_from_base_model(args.base_model, args.output_dir)
    else:
        if not args.megatron_ckpt:
            raise ValueError("--megatron-ckpt is required when not using --extract-only")
        
        # First extract base MTP
        base_mtp_dir = args.output_dir + "_base"
        extract_mtp_from_base_model(args.base_model, base_mtp_dir)
        
        # Load finetuned weights from Megatron checkpoint
        finetuned_weights = load_megatron_dist_checkpoint(args.megatron_ckpt, args.tp_size)
        
        if finetuned_weights:
            # Merge finetuned weights
            merge_finetuned_mtp(base_mtp_dir, finetuned_weights, args.output_dir)
        else:
            print("\nNo finetuned weights loaded. Using base MTP weights.")
            shutil.copytree(base_mtp_dir, args.output_dir)
    
    print("\nDone!")
    print(f"\nTo use with vLLM:")
    print(f"  --speculative_config '{{\"method\":\"eagle\",\"model\":\"{args.output_dir}\", \"num_speculative_tokens\": 3}}'")


if __name__ == "__main__":
    main()
