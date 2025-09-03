# count.py
import os
import sys
import time
from pathlib import Path
from typing import List, Any, Dict

print(">>> starting count.py", flush=True)

# --- Unbuffered stdout just in case ---
sys.stdout.reconfigure(line_buffering=True)

# =============================
# 1) Cache setup (ephemeral)
# =============================
HF_HOME = "/ephemeral/hf_cache"
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", f"{HF_HOME}/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{HF_HOME}/hub")

for p in [HF_HOME, f"{HF_HOME}/hub", f"{HF_HOME}/datasets", f"{HF_HOME}/hub/.locks"]:
    Path(p).mkdir(parents=True, exist_ok=True)

print(">>> env ready", flush=True)

# =============================
# 2) Imports (after env)
# =============================
try:
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from tqdm import tqdm
except Exception as e:
    print(f"!!! import error: {e}", flush=True)
    raise

# =============================
# 3) Config
# =============================
MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
SPLIT          = "first_stage_test"   # CAIL2018
BATCH_SIZE     = 16
MAX_NEW_TOKENS = 64
MAX_MODEL_LEN  = 4096
TP_SIZE        = 2
MAX_NUM_BATCHED_TOKENS = 8192

print(f">>> config: model={MODEL_ID}, split={SPLIT}, bs={BATCH_SIZE}, "
      f"max_len={MAX_MODEL_LEN}, tp={TP_SIZE}", flush=True)

# =============================
# 4) Helpers
# =============================
def get_accusation(ex: Dict[str, Any]) -> str:
    if "accusation" in ex:
        acc = ex["accusation"]
        if isinstance(acc, list) and acc:
            return str(acc[0])
        if isinstance(acc, str) and acc.strip():
            return acc.strip()
    if "meta" in ex:
        import json
        meta = ex["meta"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if isinstance(meta, dict):
            acc = meta.get("accusation", [])
            if isinstance(acc, list) and acc:
                return str(acc[0])
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

def truncate_to_ctx(tokenizer, prompt: str, max_model_len: int, max_new_tokens: int, reserve: int = 32) -> str:
    room = max(16, max_model_len - max_new_tokens - max(0, reserve))
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= room:
        return prompt
    kept = ids[-room:]
    return tokenizer.decode(kept, skip_special_tokens=True)

def main():
    # Load dataset
    print(">>> loading CAIL2018 …", flush=True)
    ds = load_dataset("china-ai-law-challenge/cail2018", split=SPLIT)
    print(f">>> dataset loaded: {ds}", flush=True)
    if len(ds) == 0:
        print("!!! dataset is empty — check SPLIT name", flush=True)
        return

    # Tokenizer
    print(">>> loading tokenizer …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build prompts/labels
    print(">>> building prompts …", flush=True)
    prompts: List[str] = []
    labels:  List[str] = []
    for ex in ds:
        fact  = ex.get("fact", "")
        label = get_accusation(ex)
        p     = make_prompt(fact)
        p     = truncate_to_ctx(tok, p, MAX_MODEL_LEN, MAX_NEW_TOKENS)
        prompts.append(p)
        labels.append(label)

    print(f">>> prompts ready: {len(prompts)} samples", flush=True)

    # vLLM
    print(">>> booting vLLM … (first run may take ~1 min)", flush=True)
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        tensor_parallel_size=TP_SIZE,
        trust_remote_code=True,
        download_dir=f"{HF_HOME}/hub",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    # Timed inference
    print(">>> starting timed inference …", flush=True)
    t0 = time.time()

    preds: List[str] = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating", mininterval=1.0):
        batch = prompts[i : i + BATCH_SIZE]
        outs = llm.generate(batch, sampling)
        for out in outs:
            text = (out.outputs[0].text or "").strip()
            text = text.split("\n")[0].strip("。:： ").strip()
            preds.append(text)

    t1 = time.time()
    elapsed = t1 - t0
    thr = len(prompts) / elapsed
    print(f">>> inference done in {elapsed:.2f}s, throughput {thr:.2f} samples/s", flush=True)

    # Accuracy
    correct = sum(1 for p, g in zip(preds, labels) if normalize_label(p) == normalize_label(g))
    acc = correct / len(labels)
    print(f">>> accuracy: {acc:.4f}  ({correct}/{len(labels)})", flush=True)

    # Save CSV
    try:
        import pandas as pd
        out_path = "cail_vllm_outputs.csv"
        pd.DataFrame({"prompt": prompts, "gold": labels, "pred": preds}).to_csv(out_path, index=False)
        print(f">>> saved {out_path}", flush=True)
    except Exception as e:
        print(f"!!! skip CSV save: {e}", flush=True)

    llm.shutdown()
    print(">>> done.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Force-visible error
        import traceback
        print("!!! fatal error:", flush=True)
        traceback.print_exc()
        sys.exit(1)