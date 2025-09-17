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
# 1) Cache setup (ephemeral)
# =============================
HF_HOME = "/ephemeral/hf_cache"

# 清除可能存在的旧环境变量
for key in ['HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE']:
    if key in os.environ:
        del os.environ[key]

# 设置环境变量
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = f"{HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_HOME}/hub"
os.environ["HF_DATASETS_CACHE"] = f"{HF_HOME}/datasets"

# 创建目录并设置权限
try:
    ephemeral_base = "/ephemeral"
    if not os.path.exists(ephemeral_base):
        print(f"Creating {ephemeral_base} directory...")
        os.makedirs(ephemeral_base, mode=0o755, exist_ok=True)
    
    directories = [
        HF_HOME,
        f"{HF_HOME}/hub", 
        f"{HF_HOME}/datasets",
        f"{HF_HOME}/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
        f"{HF_HOME}/datasets/china-ai-law-challenge___cail2018"
    ]
    
    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, mode=0o755, exist_ok=True)
        os.chmod(dir_path, 0o755)
    
    print(f"Successfully set up cache directories in {HF_HOME}")
    
except Exception as e:
    print(f"Error setting up directories: {e}")
    print(f"Falling back to temp directory")
    HF_HOME = os.path.join(tempfile.gettempdir(), "hf_cache")
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_CACHE"] = f"{HF_HOME}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_HOME}/hub"
    
    for p in [HF_HOME, f"{HF_HOME}/hub", f"{HF_HOME}/datasets"]:
        Path(p).mkdir(parents=True, exist_ok=True)
    print(f"Using temp cache directory: {HF_HOME}")

# =============================
# 2) Configuration
# =============================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SPLIT = "final_test"  # 可选: first_stage_train, first_stage_test, exercise_contest, final_test
BATCH_SIZE = 64  # 根据显存调整
MAX_NEW_TOKENS = 64  # 增加一些空间给复杂罪名
MAX_MODEL_LEN = 8192
TP_SIZE = 2  # 双卡并行
MAX_NUM_BATCHED_TOKENS = 16384
MAX_NUM_SEQS = 128

# 输出配置
OUTPUT_DIR = "cail_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================
# 3) Enhanced Prompt Templates
# =============================
class PromptTemplate:
    """多种prompt模板用于测试"""
    
    @staticmethod
    def basic_template(fact: str) -> str:
        """基础模板 - 简洁直接，注意输出格式"""
        messages = [
            {
                "role": "system", 
                "content": "你是法律专家，请根据案件事实判断涉嫌的罪名。\n\n注意：只输出罪名本身，不要加\"罪\"字。例如输出\"诈骗\"而不是\"诈骗罪\"。"
            },
            {
                "role": "user", 
                "content": f"案件事实：{fact.strip()}\n\n罪名："
            }
        ]
        return messages
    
    @staticmethod
    def detailed_template(fact: str) -> str:
        """详细模板 - 提供更多上下文和明确格式要求"""
        messages = [
            {
                "role": "system", 
                "content": "你是专业的刑法专家。请根据案件事实，准确判断被告人涉嫌的具体罪名。\n\n重要格式要求：\n1. 只输出罪名本身，如\"诈骗\"、\"盗窃\"、\"故意伤害\"等\n2. 不要在罪名后面加\"罪\"字\n3. 如果涉及多个罪名，输出主要罪名\n4. 不要添加任何解释、分析或其他文字"
            },
            {
                "role": "user", 
                "content": f"请根据以下案件事实判断罪名：\n\n{fact.strip()}\n\n罪名："
            }
        ]
        return messages
    
    @staticmethod
    def few_shot_template(fact: str) -> str:
        """Few-shot模板 - 提供正确格式的示例"""
        messages = [
            {
                "role": "system", 
                "content": "你是刑法专家，请根据案件事实判断罪名。注意输出格式，不要加\"罪\"字。参考以下示例："
            },
            {
                "role": "user", 
                "content": "案件事实：被告人张某在商场内趁店员不备，盗窃手机一部，价值3000元。\n罪名："
            },
            {
                "role": "assistant", 
                "content": "盗窃"
            },
            {
                "role": "user", 
                "content": "案件事实：被告人李某通过虚构事实的方式，骗取他人财物5万元。\n罪名："
            },
            {
                "role": "assistant", 
                "content": "诈骗"
            },
            {
                "role": "user", 
                "content": "案件事实：被告人王某故意伤害他人身体，致轻伤。\n罪名："
            },
            {
                "role": "assistant", 
                "content": "故意伤害"
            },
            {
                "role": "user", 
                "content": f"案件事实：{fact.strip()}\n罪名："
            }
        ]
        return messages
    
    @staticmethod
    def cot_template(fact: str) -> str:
        """Chain of Thought模板 - 要求推理过程但输出格式明确"""
        messages = [
            {
                "role": "system", 
                "content": "你是法律专家。请根据案件事实分析并判断罪名。\n\n输出格式：\n分析：[简要分析关键要素和法律条文]\n罪名：[具体罪名，不要加\"罪\"字]"
            },
            {
                "role": "user", 
                "content": f"请分析以下案件：\n\n{fact.strip()}"
            }
        ]
        return messages
    
    @staticmethod
    def structured_template(fact: str) -> str:
        """结构化模板 - 更系统的分析方法"""
        messages = [
            {
                "role": "system", 
                "content": "你是资深刑法专家。请按照以下步骤分析案件并判断罪名：\n\n1. 识别关键行为和后果\n2. 确定涉及的法律条文\n3. 判断罪名\n\n最终只输出罪名本身，不要加\"罪\"字。例如：\"诈骗\"、\"盗窃\"、\"故意伤害\"等。"
            },
            {
                "role": "user", 
                "content": f"案件事实：\n{fact.strip()}\n\n请分析并判断罪名："
            }
        ]
        return messages

# 选择使用的模板
PROMPT_TEMPLATE = PromptTemplate.structured_template  # 推荐使用few_shot模板

# =============================
# 4) Helper Functions
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
# 5) Main Execution
# =============================
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading CAIL2018 dataset...")
    ds = load_dataset("china-ai-law-challenge/cail2018", split=SPLIT)
    print(f"Dataset loaded: {len(ds)} samples")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    
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
    
    with open(f"{OUTPUT_DIR}/sample_prompts_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(sample_prompts, f, ensure_ascii=False, indent=2)
    
    print("Initializing vLLM engine...")
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        tensor_parallel_size=TP_SIZE,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
    )
    
    sampling = SamplingParams(
        temperature=0.3,  # 确定性输出
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
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Error rate: {metrics['error_rate']:.4f}")
    
    # 保存详细结果
    results = {
        "metadata": {
            "model": MODEL_ID,
            "dataset_split": SPLIT,
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "timestamp": TIMESTAMP,
            "prompt_template": PROMPT_TEMPLATE.__name__
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
    output_file = f"{OUTPUT_DIR}/cail_results_{MODEL_ID.replace('/', '_')}_{TIMESTAMP}.json"
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
        
        csv_file = f"{OUTPUT_DIR}/cail_results_{MODEL_ID.replace('/', '_')}_{TIMESTAMP}.csv"
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
    
    print("Test completed successfully!")
    return output_file, metrics

if __name__ == "__main__":
    main()