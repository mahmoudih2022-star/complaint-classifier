"""
app.py - Complaint Classifier باستخدام Claude API
"""

import os
import json
import streamlit as st
import requests

# ============================================================
# فئات الشكاوى
# ============================================================
CATEGORIES = [
    "البنية التحتية والطرق",
    "الكهرباء والمياه",
    "النظافة وجمع القمامة",
    "الصحة والمستشفيات",
    "التعليم والمدارس",
    "المواصلات والنقل",
    "الأمن والشرطة",
    "الخدمات الحكومية",
    "البيئة والتلوث",
    "الإسكان والعقارات",
    "أخرى",
]

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "") or st.secrets.get("ANTHROPIC_KEY", "")


# ============================================================
# دالة التصنيف باستخدام Claude API
# ============================================================
def classify(text: str) -> dict:
    categories_str = "\n".join([f"{i}. {c}" for i, c in enumerate(CATEGORIES)])

    prompt = f"""صنّف الشكوى التالية في إحدى الفئات المذكورة.
أجب فقط بـ JSON بالشكل ده بدون أي كلام تاني:
{{"category": "اسم الفئة", "confidence": 0.95, "top3": [{{"category": "فئة1", "confidence": 0.95}}, {{"category": "فئة2", "confidence": 0.03}}, {{"category": "فئة3", "confidence": 0.02}}]}}

الفئات المتاحة:
{categories_str}

الشكوى: {text}"""

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["content"][0]["text"].strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="تصنيف الشكاوى", page_icon="🤖", layout="centered")

# --- API mode: لما Lovable يبعت request ---
query_params = st.query_params
if "text" in query_params:
    text = query_params["text"]
    try:
        result = classify(text)
        st.json(result)
    except Exception as e:
        st.json({"error": str(e)})
    st.stop()

# --- UI mode ---
st.title("🤖 تصنيف شكاوى المواطنين")
st.markdown("اكتب الشكوى وهيصنفها الذكاء الاصطناعي تلقائياً")

text = st.text_area("نص الشكوى:", placeholder="مثال: الطريق أمام بيتنا مكسور ومحتاج إصلاح عاجل", height=120)

if st.button("صنّف الشكوى 🔍", use_container_width=True):
    if not text.strip():
        st.warning("من فضلك اكتب نص الشكوى")
    elif not ANTHROPIC_KEY:
        st.error("مفيش ANTHROPIC_KEY — أضفه في Secrets")
    else:
        with st.spinner("جاري التصنيف..."):
            try:
                result = classify(text)
                st.success(f"**الفئة:** {result['category']}")
                st.metric("نسبة الثقة", f"{result['confidence']*100:.1f}%")
                st.subheader("أعلى 3 تصنيفات:")
                for item in result["top3"]:
                    st.progress(item["confidence"], text=f"{item['category']} — {item['confidence']*100:.1f}%")
            except Exception as e:
                st.error(f"خطأ: {e}")
