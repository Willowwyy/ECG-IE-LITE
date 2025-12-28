import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
import time

# === 页面配置 ===
st.set_page_config(
    page_title="ECG 智能结构化分析系统",
    page_icon="🫀",
    layout="wide"
)

# === 1. 模型加载 (带缓存，只加载一次) ===
@st.cache_resource
def load_model():
    model_path = "/data/models/Qwen2.5-ECG-Finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return tokenizer, model

# 显示加载状态
with st.spinner('正在唤醒 AI 模型，请稍候... (首次加载约需 1 分钟)'):
    tokenizer, model = load_model()

# === 2. 推理函数 ===
def analyze_ecg(text):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Extract structured findings from the ECG report.
Output strictly in JSON format.

**Rules for Chinese Input:**
1. Translate `standard_name` to English (e.g., "房颤" -> "Atrial Fibrillation").
2. Map "急性" to `acuity: ACUTE`, "陈旧" to `acuity: CHRONIC`.

**Example:**
Input: "窦性心律。急性前壁心肌梗死。"
Output: {{"findings": [
    {{"original_text": "窦性心律", "type": "RHYTHM", "standard_name": "Sinus Rhythm", "meta": {{"location": [], "acuity": "UNKNOWN", "confidence": "HIGH"}}}},
    {{"original_text": "急性前壁心肌梗死", "type": "DIAGNOSIS", "standard_name": "Myocardial Infarction", "meta": {{"location": ["Anterior"], "acuity": "ACUTE", "confidence": "HIGH"}}}}
]}}

### Input:
{text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024, 
            temperature=0.01,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        json_str = response.split("### Response:")[-1].strip()
        data = json.loads(json_str)
        # 后处理：修复 old -> CHRONIC
        if "findings" in data:
            for item in data["findings"]:
                text_lower = item.get("original_text", "").lower()
                if "old" in text_lower and item["meta"]["acuity"] == "UNKNOWN":
                    item["meta"]["acuity"] = "CHRONIC"
        return data
    except:
        return None

# === 3. 界面布局 ===
st.title("🫀 心电报告结构化 AI 助手")
st.markdown("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📝 输入诊断报告")
    
    # 预设例子按钮
    example_text = "Sinus tachycardia. Old inferior myocardial infarction. ST depression in V4-V6."
    if st.button("使用示例 1 (心梗)"):
        st.session_state.input_text = example_text
    
    if st.button("使用示例 2 (房颤)"):
        st.session_state.input_text = "Atrial fibrillation with rapid ventricular response (130 bpm). Right bundle branch block."

    # 文本输入框
    user_input = st.text_area(
        "请输入医生写的原始文本：", 
        value=st.session_state.get("input_text", ""),
        height=200,
        placeholder="例如：Normal sinus rhythm..."
    )

    analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 结构化分析结果")
    
    if analyze_btn and user_input:
        start_time = time.time()
        result = analyze_ecg(user_input)
        end_time = time.time()
        
        if result and "findings" in result:
            st.success(f"分析完成！耗时: {end_time - start_time:.2f} 秒")
            
            # 将 JSON 转换为 DataFrame 表格展示
            df = pd.json_normalize(result["findings"])
            
            # 美化表格列名
            columns_map = {
                "type": "类型",
                "standard_name": "标准化名称",
                "original_text": "原文片段",
                "meta.location": "部位",
                "meta.acuity": "急慢性",
                "meta.confidence": "置信度"
            }
            # 只保留存在的列
            cols_to_show = [c for c in columns_map.keys() if c in df.columns]
            df_display = df[cols_to_show].rename(columns=columns_map)
            
            # 风险高亮逻辑
            def highlight_risk(row):
                # 如果是诊断(DIAGNOSIS)且不是排除(NEGATIVE)，标红
                if row["类型"] == "DIAGNOSIS" and row["置信度"] != "NEGATIVE":
                    return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_display.style.apply(highlight_risk, axis=1), 
                use_container_width=True,
                hide_index=True
            )
            
            # 展示原始 JSON (给开发者看)
            with st.expander("查看原始 JSON 数据"):
                st.json(result)
                
        else:
            st.error("解析失败，未检测到有效 JSON 输出。")

    elif not user_input:
        st.info("👈 请在左侧输入文本或选择示例。")