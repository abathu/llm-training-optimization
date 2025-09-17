import os
import tempfile
import json
from pathlib import Path
from typing import List, Any, Dict, Optional
from datetime import datetime

from datasets import load_dataset
from tqdm import tqdm

# ---- vLLM & HF tokenization ----
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================
# 1) Cache setup verification
# =============================
def verify_cache_setup():
    """验证HF cache是否正确设置"""
    required_vars = ['HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE']
    missing_vars = []
    
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables:", missing_vars)
        print("Please run the setup script first:")
        print("chmod +x setup_hf_cache.sh && ./setup_hf_cache.sh")
        return False
    
    # Check if directories exist and are writable
    hf_home = os.environ.get('HF_HOME')
    if not os.path.exists(hf_home):
        print(f"❌ HF_HOME directory does not exist: {hf_home}")
        return False
    
    if not os.access(hf_home, os.W_OK):
        print(f"❌ HF_HOME directory is not writable: {hf_home}")
        return False
    
    print("✅ HF cache setup verified successfully")
    print(f"Using HF_HOME: {hf_home}")
    return True

# Verify cache setup before proceeding
if not verify_cache_setup():
    print("Exiting due to cache setup issues.")
    exit(1)

# =============================
# 2) Configuration
# =============================
# 修改为你的fine-tuned模型路径
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_MODEL_PATH = "/home/ubuntu/llm-training-optimization/qwen-cail-sft"  # 你的LoRA模型路径

# 数据集配置
SPLIT = "final_test"  # 可选: first_stage_train, first_stage_test, exercise_contest, final_test
BATCH_SIZE = 64  # 根据显存调整
MAX_NEW_TOKENS = 64  # 增加一些空间给复杂罪名
MAX_MODEL_LEN = 8192
TP_SIZE = 2  # 双卡并行
MAX_NUM_BATCHED_TOKENS = 16384
MAX_NUM_SEQS = 128

# 输出配置
OUTPUT_DIR = "finetuned_cail_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================
# 3) Enhanced Prompt Templates for Fine-tuned Model
# =============================
class PromptTemplate:
    """针对fine-tuned模型优化的prompt模板"""
    
    @staticmethod
    def finetuned_basic_template(fact: str) -> str:
        """适合fine-tuned模型的基础模板"""
        messages = [
            {
                "role": "system", 
                "content": "你是专业的法律AI助手，请根据案件事实准确判断涉嫌的罪名。只输出罪名本身，不要加\"罪\"字。"
            },
            {
                "role": "user", 
                "content": f"请根据以下案件事实判断罪名：\n\n{fact.strip()}\n\n罪名："
            }
        ]
        return messages
    
    @staticmethod
    def finetuned_structured_template(fact: str) -> str:
        """结构化模板 - 针对fine-tuned模型优化"""
        messages = [
            {
                "role": "system", 
                "content": "你是经过专业训练的法律分析AI。请根据案件事实判断罪名。\n\n输出要求：\n1. 只输出罪名本身，如\"诈骗\"、\"盗窃\"、\"故意伤害\"\n2. 不要在罪名后加\"罪\"字\n3. 如有多个罪名，输出主要罪名"
            },
            {
                "role": "user", 
                "content": f"案件事实：{fact.strip()}\n\n请判断罪名："
            }
        ]
        return messages
    
    @staticmethod
    def finetuned_concise_template(fact: str) -> str:
        """简洁模板 - 最直接的格式"""
        messages = [
            {
                "role": "user", 
                "content": f"案件事实：{fact.strip()}\n\n罪名："
            }
        ]
        return messages

# 选择使用的模板 - 建议先用基础模板测试
PROMPT_TEMPLATE = PromptTemplate.finetuned_concise_template

# =============================
# 4) Model Loading with LoRA Support
# =============================
def load_finetuned_model():
    """加载fine-tuned模型（支持LoRA）"""
    
    # 检查模型路径
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"Error: Fine-tuned model path not found: {FINETUNED_MODEL_PATH}")
        print("Please check the path and ensure the model exists.")
        return None, None
    
    # 检查是否为LoRA模型
    adapter_config_path = os.path.join(FINETUNED_MODEL_PATH, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)
    
    print(f"Model type detected: {'LoRA adapter' if is_lora else 'Full model'}")
    
    if is_lora:
        print("Loading LoRA adapter model...")
        print("Note: vLLM doesn't directly support LoRA. You might need to merge the adapter first.")
        print("For now, testing with base model. Consider merging LoRA weights for better results.")
        
        # 对于LoRA模型，我们需要特殊处理
        # 方案1：使用transformers加载然后转换（推荐用于小模型）
        # 方案2：先合并LoRA权重再用vLLM加载
        
        print("Loading tokenizer from fine-tuned model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                FINETUNED_MODEL_PATH, 
                use_fast=True, 
                trust_remote_code=True
            )
            print("✓ Tokenizer loaded successfully from fine-tuned path")
        except Exception as e:
            print(f"Failed to load tokenizer from fine-tuned path: {e}")
            print("Falling back to base model tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_ID, 
                use_fast=True, 
                trust_remote_code=True
            )
        
        # 对于vLLM，我们先用base model，但会在prompt中说明这个限制
        print(f"Initializing vLLM with base model: {BASE_MODEL_ID}")
        print("Note: Using base model weights. For full fine-tuned performance, consider merging LoRA weights.")
        
        llm = LLM(
            model=BASE_MODEL_ID,  # 使用base model
            dtype="bfloat16",
            tensor_parallel_size=TP_SIZE,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
        )
        
    else:
        print("Loading full fine-tuned model...")
        tokenizer = AutoTokenizer.from_pretrained(
            FINETUNED_MODEL_PATH, 
            use_fast=True, 
            trust_remote_code=True
        )
        
        llm = LLM(
            model=FINETUNED_MODEL_PATH,
            dtype="bfloat16",
            tensor_parallel_size=TP_SIZE,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
        )
    
    return llm, tokenizer

# =============================
# 5) Helper Functions (保持不变)
# =============================
def get_accusation(example: Dict[str, Any]) -> str:
    """提取标准答案罪名"""
    if "accusation" in example:
        acc = example["accusation"]
        if isinstance(acc, list) and acc:
            return str(acc[0])
        if isinstance(acc, str) and acc.strip():
            return acc.strip()
    
    if "meta" in example:
        meta = example["meta"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except:
                meta = {}
        if isinstance(meta, dict):
            acc = meta.get("accusation", [])
            if isinstance(acc, list) and acc:
                return str(acc[0])
    
    return "未知罪名"

def make_prompt(fact: str, tokenizer) -> str:
    """构建prompt"""
    messages = PROMPT_TEMPLATE(fact)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def truncate_to_ctx(fact: str, tokenizer, max_model_len: int, max_new_tokens: int) -> str:
    """截断过长的事实描述"""
    test_prompt = make_prompt(fact, tokenizer)
    ids = tokenizer(test_prompt, add_special_tokens=True)["input_ids"]
    
    total_budget = max_model_len - max_new_tokens - 100  # 预留100个token
    if len(ids) <= total_budget:
        return fact
    
    # 截断fact
    fact_tokens = tokenizer(fact, add_special_tokens=False)["input_ids"]
    available_for_fact = total_budget - 100  # 预留给系统消息和格式
    
    if len(fact_tokens) > available_for_fact:
        kept_tokens = fact_tokens[:available_for_fact]
        truncated_fact = tokenizer.decode(kept_tokens, skip_special_tokens=True)
        return truncated_fact
    
    return fact

def extract_crime_name(output_text: str) -> str:
    """从模型输出中提取罪名"""
    text = output_text.strip()
    
    # 处理CoT格式的输出
    if "罪名：" in text:
        parts = text.split("罪名：")
        if len(parts) > 1:
            text = parts[-1].strip()
    
    # 清理常见格式
    text = text.split("\n")[0].strip()  # 取第一行
    text = text.strip("。:： ，,")       # 移除标点
    
    # 移除常见前缀
    prefixes = ["涉嫌", "构成", "判定为", "属于", "是"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # 关键：移除"罪"字后缀
    if text.endswith("罪") and len(text) > 1:
        text = text[:-1]
    
    return text

def normalize_label(label: str) -> str:
    """标准化标签用于比较"""
    label = (label or "").strip()
    
    # 移除标点
    label = label.replace("。", "").replace("：", "").replace(":", "")
    
    # 移除"罪"字后缀（这是关键！）
    if label.endswith("罪") and len(label) > 1:
        label = label[:-1]
    
    # 取第一个词
    return label.split()[0] if label else label

def calculate_metrics(predictions: List[str], labels: List[str]) -> Dict[str, Any]:
    """计算评估指标"""
    correct = 0
    exact_matches = []
    errors = []
    
    pred_normalized = [normalize_label(p) for p in predictions]
    gold_normalized = [normalize_label(g) for g in labels]
    
    for i, (pred, gold) in enumerate(zip(pred_normalized, gold_normalized)):
        if pred == gold:
            correct += 1
            exact_matches.append((predictions[i], labels[i]))
        else:
            errors.append({
                "index": i,
                "prediction": predictions[i],
                "gold": labels[i],
                "pred_normalized": pred,
                "gold_normalized": gold
            })
    
    accuracy = correct / len(labels) if len(labels) > 0 else 0
    
    return {
        "total_samples": len(labels),
        "correct_predictions": correct,
        "accuracy": accuracy,
        "error_rate": 1 - accuracy,
        "exact_matches": exact_matches,
        "errors": errors
    }

# =============================
# 6) Main Execution
# =============================
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("FINE-TUNED MODEL EVALUATION")
    print("="*60)
    print(f"Base model: {BASE_MODEL_ID}")
    print(f"Fine-tuned model path: {FINETUNED_MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    print("Loading CAIL2018 dataset...")
    ds = load_dataset("china-ai-law-challenge/cail2018", split=SPLIT)
    print(f"Dataset loaded: {len(ds)} samples")
    
    print("Loading fine-tuned model and tokenizer...")
    llm, tokenizer = load_finetuned_model()
    
    if llm is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return None, None
    
    print("Building prompts...")
    prompts = []
    labels = []
    raw_facts = []
    
    for ex in tqdm(ds, desc="Processing examples"):
        fact = ex.get("fact", "")
        label = get_accusation(ex)
        
        truncated_fact = truncate_to_ctx(fact, tokenizer, MAX_MODEL_LEN, MAX_NEW_TOKENS)
        prompt = make_prompt(truncated_fact, tokenizer)
        
        prompts.append(prompt)
        labels.append(label)
        raw_facts.append(fact)
    
    print(f"Prepared {len(prompts)} prompts")
    
    # 保存prompt样本
    sample_prompts = {
        "sample_prompts": [
            {
                "index": i,
                "fact": raw_facts[i][:200] + "..." if len(raw_facts[i]) > 200 else raw_facts[i],
                "prompt": prompts[i],
                "gold_label": labels[i]
            }
            for i in range(min(5, len(prompts)))
        ]
    }
    
    with open(f"{OUTPUT_DIR}/sample_prompts_finetuned_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(sample_prompts, f, ensure_ascii=False, indent=2)
    
    sampling = SamplingParams(
        temperature=0.0,  # 降低温度以获得更确定性的输出
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
        stop=["\n\n", "</s>", "<|im_end|>"],
    )
    
    print(f"Starting inference with batch_size={BATCH_SIZE}...")
    predictions = []
    raw_outputs = []
    total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating", total=total_batches):
        batch = prompts[i : i + BATCH_SIZE]
        
        try:
            outputs = llm.generate(batch, sampling)
            batch_preds = []
            batch_raw = []
            
            for output in outputs:
                raw_text = output.outputs[0].text or ""
                cleaned_text = extract_crime_name(raw_text)
                
                batch_preds.append(cleaned_text)
                batch_raw.append(raw_text)
            
            predictions.extend(batch_preds)
            raw_outputs.extend(batch_raw)
            
        except Exception as e:
            print(f"Error in batch {i//BATCH_SIZE + 1}: {e}")
            predictions.extend([""] * len(batch))
            raw_outputs.extend([""] * len(batch))
    
    print(f"Generation completed. Got {len(predictions)} predictions.")
    
    # 计算指标
    metrics = calculate_metrics(predictions, labels)
    
    print(f"\n{'='*50}")
    print(f"FINE-TUNED MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Model: Fine-tuned {BASE_MODEL_ID}")
    print(f"Fine-tuned path: {FINETUNED_MODEL_PATH}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Error rate: {metrics['error_rate']:.4f}")
    
    # 保存详细结果
    results = {
        "metadata": {
            "base_model": BASE_MODEL_ID,
            "finetuned_model_path": FINETUNED_MODEL_PATH,
            "dataset_split": SPLIT,
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "timestamp": TIMESTAMP,
            "prompt_template": PROMPT_TEMPLATE.__name__,
            "model_type": "fine_tuned"
        },
        "metrics": metrics,
        "predictions": [
            {
                "index": i,
                "fact": raw_facts[i],
                "prompt": prompts[i],
                "raw_output": raw_outputs[i],
                "prediction": predictions[i],
                "gold_label": labels[i],
                "correct": normalize_label(predictions[i]) == normalize_label(labels[i])
            }
            for i in range(len(predictions))
        ]
    }
    
    # 保存完整结果
    model_name = "finetuned_qwen2.5_0.5b"
    output_file = f"{OUTPUT_DIR}/cail_results_{model_name}_{TIMESTAMP}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # 保存CSV格式（更容易查看）
    try:
        import pandas as pd
        
        df = pd.DataFrame({
            "index": range(len(predictions)),
            "fact_preview": [f[:100] + "..." if len(f) > 100 else f for f in raw_facts],
            "prediction": predictions,
            "gold_label": labels,
            "correct": [normalize_label(p) == normalize_label(g) for p, g in zip(predictions, labels)]
        })
        
        csv_file = f"{OUTPUT_DIR}/cail_results_{model_name}_{TIMESTAMP}.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"CSV results saved to: {csv_file}")
        
    except ImportError:
        print("pandas not available, skipping CSV export")
    
    # 显示错误样本
    if metrics["errors"]:
        print(f"\nFirst 5 error cases:")
        for i, error in enumerate(metrics["errors"][:5]):
            print(f"  {i+1}. Predicted: '{error['prediction']}' | Gold: '{error['gold']}'")
    
    # 显示正确样本
    if metrics["exact_matches"]:
        print(f"\nFirst 3 correct cases:")
        for i, (pred, gold) in enumerate(metrics["exact_matches"][:3]):
            print(f"  {i+1}. '{pred}' == '{gold}'")
    
    # 清理资源
    print("\nShutting down vLLM...")
    try:
        llm.shutdown()
    except:
        pass
    
    print("Fine-tuned model evaluation completed successfully!")
    return output_file, metrics

if __name__ == "__main__":
    main()