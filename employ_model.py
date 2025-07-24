import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# =====================
# Step 1: Setup HPC Cache Path
# =====================
os.environ["HF_HOME"] = "/home/s2678328/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/s2678328/.cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/home/s2678328/.cache/datasets"

# =====================
# Step 2: Load Dataset
# =====================
print("Loading dataset...")
dataset = load_dataset("lambada", "plain_text", split="validation", cache_dir="/home/s2678328/.cache/datasets")

# For demo, use a subset (e.g., first 50 samples)
dataset = dataset.select(range(50))

# =====================
# Step 3: Load Model and Tokenizer
# =====================
model_path = "/home/s2678328/llm-training-optimization/model_cache/Qwen3-14B"
print(f"Loading model from {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,   # Use FP16 for efficiency
    device_map="auto",           # Automatically split layers across GPUs
)

# =====================
# Step 4: Helper Function: Prepare Input
# =====================
def prepare_input(prompt_text):
    """Format dataset text into chat template for Qwen3 models"""
    messages = [{"role": "user", "content": prompt_text}]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return chat_text

# =====================
# Step 5: Batched Inference Loop
# =====================
batch_size = 4   # Adjust for memory; 4 prompts per batch
results = []

print("Running inference...")
for i in tqdm(range(0, len(dataset), batch_size)):
    # Prepare batch
    batch_prompts = [prepare_input(text) for text in dataset[i:i+batch_size]["text"]]
    batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**batch_inputs, max_new_tokens=512)

    # Process each sample in batch
    for j, gen_ids in enumerate(generated_ids):
        # Extract only generated tokens
        prompt_len = len(batch_inputs.input_ids[j])
        output_ids = gen_ids[prompt_len:].tolist()

        # Find </think> token (151668)
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        results.append({
            "input": dataset[i + j]["text"],
            "thinking": thinking_content,
            "output": content
        })

# =====================
# Step 6: Save Outputs
# =====================
output_file = "inference_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_file}")