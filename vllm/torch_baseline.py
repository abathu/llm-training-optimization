#!/usr/bin/env python3
import os
import time
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm   # ✅ added

# -----------------------------
# Helpers
# -----------------------------
def build_cache_dirs(hf_home: str):
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", f"{hf_home}/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{hf_home}/hub")
    for p in [hf_home, f"{hf_home}/hub", f"{hf_home}/datasets", f"{hf_home}/hub/.locks"]:
        Path(p).mkdir(parents=True, exist_ok=True)

def auto_max_memory(headroom_gib: int = 3) -> Dict[Any, str]:
    n = torch.cuda.device_count()
    if n == 0:
        return {"cpu": "64GiB"}
    mm = {}
    for i in range(n):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        cap = max(1, int(total_gib) - headroom_gib)
        mm[i] = f"{cap}GiB"
    mm["cpu"] = "64GiB"
    return mm

def get_accusation(ex: Dict[str, Any]) -> str:
    if "accusation" in ex:
        a = ex["accusation"]
        if isinstance(a, list) and a:
            return str(a[0])
        if isinstance(a, str) and a.strip():
            return a.strip()
    if "meta" in ex:
        meta = ex["meta"]
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if isinstance(meta, dict):
            a = meta.get("accusation", [])
            if isinstance(a, list) and a:
                return str(a[0])
    return "未知罪名"

def make_prompt(fact: str) -> str:
    return (
        "你是法律助理。根据下述案件事实，判断罪名并仅输出罪名词条，不要输出其它解释。\n"
        f"案件事实：{(fact or '').strip()}\n"
        "答："
    )

def normalize_label(s: str) -> str:
    s = (s or "").strip().replace("。", "").replace("：", ":")
    if "\n" in s:
        s = s.split("\n", 1)[0]
    if ":" in s:
        s = s.split(":", 1)[-1].strip()
    return s.split()[0] if s else s

def truncate_prompt(tokenizer, prompt: str, max_model_len: int, max_new_tokens: int, reserve: int = 32) -> str:
    room = max(16, max_model_len - max_new_tokens - max(0, reserve))
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= room:
        return prompt
    kept = ids[-room:]
    return tokenizer.decode(kept, skip_special_tokens=True)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="PyTorch baseline inference on CAIL2018")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--split", default="first_stage_test")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--hf-home", default="/ephemeral/hf_cache")
    parser.add_argument("--headroom-gib", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_cache_dirs(args.hf_home)

    n_gpus = torch.cuda.device_count()
    print(f"GPUs visible: {n_gpus}")
    assert n_gpus >= 1, "Need at least 1 GPU."

    print(f"Loading dataset: china-ai-law-challenge/cail2018 [{args.split}] …")
    ds = load_dataset("china-ai-law-challenge/cail2018", split=args.split)

    print(f"Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    max_memory = auto_max_memory(headroom_gib=args.headroom_gib)
    print("max_memory map:", max_memory)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="balanced",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    prompts, labels = [], []
    for ex in ds:
        fact = ex.get("fact", "")
        gold = get_accusation(ex)
        p = make_prompt(fact)
        p = truncate_prompt(tokenizer, p, args.max_model_len, args.max_new_tokens)
        prompts.append(p)
        labels.append(gold)

    torch.manual_seed(args.seed)
    total_start = time.time()

    preds: List[str] = []
    total_tokens = 0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    num_batches = math.ceil(len(prompts) / args.batch)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch_dtype):
        for i in tqdm(range(0, len(prompts), args.batch), total=num_batches, desc="Generating"):
            batch_prompts = prompts[i:i + args.batch]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_model_len - args.max_new_tokens - 32,
            )
            for k in enc:
                enc[k] = enc[k].cuda()

            out = model.generate(**enc, **gen_kwargs)
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            input_lens = (enc["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()

            for t, in_len in zip(texts, input_lens):
                pred = tokenizer.decode(
                    tokenizer(t, add_special_tokens=False)["input_ids"][in_len:],
                    skip_special_tokens=True
                ).strip()
                pred = pred.split("\n")[0].strip("。:： ").strip()
                preds.append(pred)

            total_tokens += sum(input_lens) + len(out[0]) - input_lens[0]

    total_time = time.time() - total_start
    correct = sum(1 for p, g in zip(preds, labels) if normalize_label(p) == normalize_label(g))
    acc = correct / len(labels)

    toks_per_sec = total_tokens / total_time if total_time > 0 else float("nan")
    print("\n===== Inference Summary (PyTorch baseline) =====")
    print(f"Samples               : {len(labels)}")
    print(f"Batch size            : {args.batch}")
    print(f"Max model len         : {args.max_model_len}")
    print(f"Max new tokens        : {args.max_new_tokens}")
    print(f"Total wall time (s)   : {total_time:.2f}")
    print(f"Throughput (tok/s)    : {toks_per_sec:.2f}")
    print(f"Exact-match accuracy  : {acc:.4f}  ({correct}/{len(labels)})")
    print("================================================")

if __name__ == "__main__":
    main()