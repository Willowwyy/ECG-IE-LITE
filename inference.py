import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import argparse
import sys

# === 核心推理函数 (封装以便复用) ===
def load_model(model_path):
    print(f"正在加载模型: {model_path} ...")
    st = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",      
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        print(f"模型加载失败，请检查路径: {e}")
        sys.exit(1)
        
    print(f"模型加载完成！耗时: {time.time() - st:.2f}s")
    return tokenizer, model

def analyze_report(model, tokenizer, report_text):
    # 构造与训练时一致的 Alpaca 格式 Prompt
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Extract structured findings from the ECG report.
Output strictly in JSON format.

**Rules for Chinese Input:**
1. Translate `standard_name` to English (e.g., "房颤" -> "Atrial Fibrillation").
2. Map "急性" to `acuity: ACUTE`, "陈旧" to `acuity: CHRONIC`.

### Input:
{report_text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,
            temperature=0.01, # 趋近于 0，让输出最稳定
            top_p=0.9
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 JSON 部分并进行简单的后处理
    try:
        json_str = response.split("### Response:")[-1].strip()
        # 处理可能的 Markdown 代码块标记
        if json_str.startswith("```json"):
            json_str = json_str[7:-3]
            
        data = json.loads(json_str)
        
        # Rule-based Fix: 修复 "old" -> "CHRONIC"
        if "findings" in data:
            for item in data["findings"]:
                text_lower = item.get("original_text", "").lower()
                if "old" in text_lower and item.get("meta", {}).get("acuity") == "UNKNOWN":
                    item["meta"]["acuity"] = "CHRONIC"
                    
        return data
    except Exception as e:
        print(f"JSON 解析警告: {e}")
        return {"raw_output": response}

# === 主程序入口 ===
if __name__ == "__main__":
    # 使用 argparse 让路径可配置，体现工程化思维
    parser = argparse.ArgumentParser(description="ECG-IE-Lite Inference CLI")
    parser.add_argument("--model_path", type=str, default="/data/models/Qwen2.5-ECG-Finetuned", help="Path to the finetuned model or adapter")
    args = parser.parse_args()

    # 1. 加载模型
    tokenizer, model = load_model(args.model_path)

    # 2. 准备测试用例
    test_cases = [
        "Sinus tachycardia. Old inferior myocardial infarction. ST depression in V4-V6.",
        "Atrial fibrillation with rapid ventricular response (130 bpm). Right bundle branch block.",
        "Normal sinus rhythm. No ST-T abnormalities.",
        "窦性心律。急性前壁心肌梗死。" # 测试一下中文输入（如果在训练集中有的话）
    ]

    print("\n" + "="*50)
    print("🚀 开始批量推理测试 (CLI Mode)")
    print("="*50)

    for i, case in enumerate(test_cases):
        print(f"\n[Case {i+1}] 输入报告: {case}")
        result = analyze_report(model, tokenizer, case)
        
        print(f"[AI 解析结果]:")
        # ensure_ascii=False 保证中文正常显示
        print(json.dumps(result, indent=2, ensure_ascii=False)) 
        print("-" * 50)