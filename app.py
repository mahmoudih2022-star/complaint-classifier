"""
app.py - Complaint Classifier على Streamlit Cloud
"""

import os
import streamlit as st
import requests
import json

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

HF_TOKEN = os.environ.get("HF_TOKEN", st.secrets.get("HF_TOKEN", ""))
MODEL    = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # يدعم العربية ✅


# ============================================================
# دالة التصنيف
# ============================================================
def classify(text: str) -> dict:
    url     = f"https://api-inference.huggingface.co/models/{MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": CATEGORIES},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    labels = data.get("labels", [])
    scores = data.get("scores", [])

    top3 = [
        {"category": labels[i], "confidence": round(scores[i], 4)}
        for i in range(min(3, len(labels)))
    ]

    return {
        "category":   labels[0] if labels else "أخرى",
        "confidence": round(scores[0], 4) if scores else 0.0,
        "top3":       top3,
    }


# ============================================================
# Streamlit UI + REST endpoint
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

# --- UI mode: واجهة اختبار ---
st.title("🤖 تصنيف شكاوى المواطنين")
st.markdown("اكتب الشكوى وهيصنفها الذكاء الاصطناعي تلقائياً")

text = st.text_area("نص الشكوى:", placeholder="مثال: الطريق أمام بيتنا مكسور ومحتاج إصلاح عاجل", height=120)

if st.button("صنّف الشكوى 🔍", use_container_width=True):
    if not text.strip():
        st.warning("من فضلك اكتب نص الشكوى")
    elif not HF_TOKEN:
        st.error("مفيش HF_TOKEN — أضفه في secrets")
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
