import json
import torch
import time  # 引入时间库
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 配置 ===
FINETUNED_PATH = "/data/models/Qwen2.5-ECG-Finetuned"
TEST_FILE = "data/test.jsonl"
NUM_SAMPLES = 50 

# === 1. 加载模型 ===
print("正在加载微调模型...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH)
model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_PATH, device_map="auto", torch_dtype=torch.bfloat16
)

def run_inference(model, tokenizer, text):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Extract structured findings from the ECG report.

### Input:
{text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # === ⏱️ 开始计时 ===
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.01 
        )
    
    # === ⏱️ 结束计时 ===
    end_time = time.time()
    latency = end_time - start_time # 计算耗时 (秒)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        json_obj = json.loads(response.split("### Response:")[-1].strip())
    except:
        json_obj = {"findings": []}
        
    return json_obj, latency  # 返回结果和耗时

# === 评分函数 (保持不变) ===
def calculate_f1(pred_json, true_json):
    def get_findings(data):
        findings = set()
        if "findings" in data:
            for item in data["findings"]:
                name = item.get("standard_name", "").lower().strip()
                if name: findings.add(name)
        return findings

    pred_set = get_findings(pred_json)
    true_set = get_findings(true_json)
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# === 2. 开始测试 ===
f1_list = []
latency_list = [] # 用于存每一条的时间

print(f"🚀 开始评测微调模型 (Acc & Speed)...")

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()[:NUM_SAMPLES]

for line in tqdm(lines):
    entry = json.loads(line)
    input_text = entry['input']
    true_output = json.loads(entry['output'])
    
    # 运行推理 (带测速)
    pred_output, latency = run_inference(model, tokenizer, input_text)
    
    # 记录数据
    f1 = calculate_f1(pred_output, true_output)
    f1_list.append(f1)
    latency_list.append(latency)

# === 3. 计算最终统计数据 ===
avg_f1 = np.mean(f1_list)
avg_latency = np.mean(latency_list)

print("\n" + "="*40)
print(f"📊 你的模型 (Ours) 最终成绩单:")
print(f"✅ 平均 F1-Score: {avg_f1:.4f}")
print(f"⚡ 平均推理耗时: {avg_latency:.4f} 秒/条")
print("="*40)

# === 4. 自动画图 (填入你提供的真实数据) ===
# 这里的 0 和 0.3639 是你刚才测出来的 Base 和 DeepSeek 的分
# 这里的 3.1 和 7.68 是你刚才测出来的 Base 和 DeepSeek 的速度
models = ['Base Model', 'DeepSeek (API)', 'Ours (Finetuned)']
scores = [0.0, 0.3639, avg_f1]          # 准确率对比
times  = [3.1, 7.68, avg_latency]       # 速度对比 (用实测值)

fig, ax1 = plt.subplots(figsize=(10, 6))

# 柱状图：F1 Score
color = 'tab:blue'
ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1 Accuracy Score', color=color, fontsize=12)
bars = ax1.bar(models, scores, color=['gray', '#4c72b0', '#55a868'], alpha=0.7, width=0.5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1.1)

# 标数值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 折线图：速度
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Inference Latency (seconds)', color=color, fontsize=12)
ax2.plot(models, times, color=color, marker='o', linestyle='-', linewidth=2, markersize=8)
ax2.tick_params(axis='y', labelcolor=color)

# 标速度数值
for i, txt in enumerate(times):
    ax2.annotate(f"{txt:.2f}s", (i, times[i]), xytext=(0, 10), 
                 textcoords='offset points', ha='center', color='red', fontweight='bold')

plt.title('Comparison: Accuracy vs. Speed', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.savefig('final_comparison_result.png', dpi=300)
print("\n🖼️ 对比图已保存为: final_comparison_result.png")