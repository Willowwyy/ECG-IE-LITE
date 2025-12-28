import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

# === 1. 配置：指向原始底座模型路径 ===
# 没有微调过的模型
BASE_MODEL_PATH = "/data/models/Qwen2.5-7B-Instruct"
TEST_FILE = "data/test.jsonl"
NUM_SAMPLES = 50 

print("正在加载原始基座模型 (Base Model)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

# === 2. 评分逻辑  ===
def calculate_f1(pred_json, true_json):
    def get_findings(data):
        findings = set()
        if isinstance(data, dict) and "findings" in data:
            for item in data["findings"]:
                name = item.get("standard_name", "").lower().strip()
                if name: findings.add(name)
        return findings
    
    pred = get_findings(pred_json)
    true_set = get_findings(true_json)
    
    tp = len(pred.intersection(true_set))
    fp = len(pred - true_set)
    fn = len(true_set - pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# === 3. 开始考试 ===
scores = []
print(f"开始测试 Base Model 前 {NUM_SAMPLES} 条...")

with open(TEST_FILE, 'r') as f:
    lines = f.readlines()[:NUM_SAMPLES]

for line in tqdm(lines):
    entry = json.loads(line)
    input_text = entry['input']
    true_output = json.loads(entry['output'])
    
    prompt = f"""Extract structured findings from the ECG report. Output strictly in JSON format.
Input: {input_text}
Response:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.01)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 尝试解析（原始模型经常输出乱七八糟的东西，所以解析失败率很高）
    try:
        json_str = response.split("Response:")[-1].strip()
        # 清理一下可能的 markdown 符号
        if json_str.startswith("```json"):
            json_str = json_str[7:-3]
        pred_output = json.loads(json_str)
    except:
        pred_output = {} # 解析失败，得0分
    
    scores.append(calculate_f1(pred_output, true_output))

print(f"\n📊 原始基座模型 (Base) 平均 F1: {np.mean(scores):.4f}")