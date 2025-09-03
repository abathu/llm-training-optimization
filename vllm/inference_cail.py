import os
from pathlib import Path
from typing import List, Any, Dict

from datasets import load_dataset
from tqdm import tqdm

# ---- vLLM & HF tokenization ----
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================
# 1) Cache setup (ephemeral)
# =============================
HF_HOME = "/ephemeral/hf_cache"
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", f"{HF_HOME}/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{HF_HOME}/hub")

# make sure the dirs exist and are writable
for p in [HF_HOME, f"{HF_HOME}/hub", f"{HF_HOME}/datasets", f"{HF_HOME}/hub/.locks"]:
    Path(p).mkdir(parents=True, exist_ok=True)

# =============================
# 2) Config
# =============================
MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"      # swap to your tuned/merged path if needed
SPLIT          = "first_stage_test"              # CAIL2018: first_stage_train | first_stage_test | exercise_* | final_test
BATCH_SIZE     = 16
MAX_NEW_TOKENS = 64
MAX_MODEL_LEN  = 4096                            # raise to 8192/16384 if you have VRAM headroom
TP_SIZE        = 2                               # tensor_parallel_size (2x A6000)

# Optional guardrail so one batch can't explode memory with very long prompts
# (set to None to let vLLM decide)
MAX_NUM_BATCHED_TOKENS = 8192

# =============================
# 3) Helpers for CAIL labels
# =============================
def get_accusation(example: Dict[str, Any]) -> str:
    """Prefer 'accusation' (list/str); else try 'meta'. Fallback to a neutral label."""
    if "accusation" in example:
        acc = example["accusation"]
        if isinstance(acc, list) and acc:
            return str(acc[0])
        if isinstance(acc, str) and acc.strip():
            return acc.strip()
    if "meta" in example:
        meta = example["meta"]
        if isinstance(meta, str):
            import json
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

# =============================
# 4) Load dataset & tokenizer
# =============================
print("Loading CAIL2018 …")
ds = load_dataset("china-ai-law-challenge/cail2018", split=SPLIT)
print(ds)

print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)

def truncate_to_ctx(prompt: str, max_model_len: int, max_new_tokens: int, reserve: int = 32) -> str:
    """
    Ensure prompt tokens + generated tokens + a small reserve <= max_model_len.
    Keep the *most recent* tokens (typical for LLM prompts).
    """
    room = max(16, max_model_len - max_new_tokens - max(0, reserve))
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= room:
        return prompt
    kept = ids[-room:]  # keep tail
    return tokenizer.decode(kept, skip_special_tokens=True)

# =============================
# 5) Build prompts & gold labels
# =============================
prompts: List[str] = []
labels:  List[str] = []

for ex in ds:
    fact  = ex.get("fact", "")
    label = get_accusation(ex)
    p     = make_prompt(fact)
    p     = truncate_to_ctx(p, MAX_MODEL_LEN, MAX_NEW_TOKENS)  # <— avoid vLLM length errors
    prompts.append(p)
    labels.append(label)

# =============================
# 6) Create vLLM engine
# =============================
print("Booting vLLM (first time can take ~1 minute)…")
llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    tensor_parallel_size=TP_SIZE,
    trust_remote_code=True,
    download_dir=f"{HF_HOME}/hub",
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=0.88,
    # enforce_eager improves stability on some drivers (slightly slower than CUDA Graph)
    enforce_eager=True,
    # cap batch token pile-up to avoid surprises (optional)
    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
)

sampling = SamplingParams(
    temperature=0.0,      # deterministic for eval
    top_p=1.0,
    max_tokens=MAX_NEW_TOKENS,
)

# =============================
# 7) Batched generation
# =============================
preds: List[str] = []

for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating"):
    batch = prompts[i : i + BATCH_SIZE]
    outs = llm.generate(batch, sampling)
    for out in outs:
        text = (out.outputs[0].text or "").strip()
        text = text.split("\n")[0].strip("。:： ").strip()
        preds.append(text)

# =============================
# 8) Simple exact-match accuracy
# =============================
assert len(preds) == len(labels)
correct = sum(1 for p, g in zip(preds, labels) if normalize_label(p) == normalize_label(g))
acc = correct / len(labels)

print(f"\nSamples: {len(labels)}")
print(f"Exact-match accuracy: {acc:.4f}  ({correct}/{len(labels)})")

# =============================
# 9) Optional: save CSV
# =============================
try:
    import pandas as pd
    out_path = "cail_vllm_outputs.csv"
    pd.DataFrame({"prompt": prompts, "gold": labels, "pred": preds}).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
except Exception as e:
    print(f"(Skip saving CSV) {e}")

# Clean shutdown avoids noisy teardown logs
llm.shutdown()
print("Done.")