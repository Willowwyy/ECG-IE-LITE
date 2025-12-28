import json
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区 =================
# 请确保这里填的是真实的 Key
API_KEY = "*********"
BASE_URL = "https://api.deepseek.com"
# 确保使用绝对路径，避免找不到文件
TEST_FILE = "data/test.jsonl"
NUM_SAMPLES = 50
# =========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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

# === 核心：完整的“考试大纲” ===
FULL_SYSTEM_PROMPT = """
You are an expert Cardiologist. Extract structured findings from the ECG report.
Output strictly in JSON format.

### Schema Definitions
1. **type**: `RHYTHM`, `MORPHOLOGY`, `DIAGNOSIS`
2. **standard_name**: The normalized clinical term. 
   - CRITICAL: Remove locations (e.g., "Anterior MI" -> "Myocardial Infarction").
   - KEEP severity (e.g., "First Degree AV Block").
3. **meta**:
   - `location`: List of locations.
   - `acuity`: "ACUTE", "CHRONIC", "UNKNOWN".
   - `confidence`: "HIGH", "LOW", "NEGATIVE".
"""

scores = []
print(f"🚀 开始测试 DeepSeek (Teacher) 前 {NUM_SAMPLES} 条数据...")

# 使用 utf-8 编码读取，防止报错
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()[:NUM_SAMPLES]

for line in tqdm(lines, desc="评测进度"):
    entry = json.loads(line)
    input_text = entry['input']
    true_output = json.loads(entry['output'])
    
    try:
        # 调用 API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": FULL_SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
        pred_str = response.choices[0].message.content
        pred_output = json.loads(pred_str)
        
        # 评分
        f1 = calculate_f1(pred_output, true_output)
        scores.append(f1)

    except Exception as e:
        print(f"⚠️ 某条数据请求失败: {e}")
        scores.append(0)

# 计算最终平均分
final_score = np.mean(scores)
print(f"\n" + "="*40)
print(f"📊 DeepSeek (Teacher) 最终平均 F1: {final_score:.4f}")
print(f"="*40)