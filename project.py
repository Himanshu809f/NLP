import streamlit as st
from transformers import pipeline

# Models load केवल एक बार होंगे
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment = pipeline("sentiment-analysis")
    return summarizer, sentiment

summarizer, sentiment_analyzer = load_models()

st.title("📝 Summarizer & Sentiment Analyzer")
st.write("नीचे टेक्स्ट पेस्ट करें और सारांश (summary) तथा भाव (sentiment) रिपोर्ट पाएं।")

text = st.text_area("सारांश के लिए टेक्स्ट:")

max_len = st.slider("अधिकतम सारांश लंबाई", 20, 200, 100)
min_len = st.slider("न्यूनतम सारांश लंबाई", 10, 100, 30)

if st.button("विश्लेषण करें"):
    if text.strip():
        # सारांश निकालना
        summary = summarizer(
            text, 
            max_length=max_len, 
            min_length=min_len, 
            do_sample=False
        )[0]["summary_text"]

        # भाव विश्लेषण
        sentiment_result = sentiment_analyzer(text[:512])  # पहले 512 characters तक
        sentiment_label = sentiment_result[0]["label"]
        sentiment_score = sentiment_result[0]["score"]

        # परिणाम दिखाना
        st.subheader("📌 सारांश")
        st.write(summary)

        st.subheader("📊 भाव विश्लेषण")
        st.json({
            "लेबल": sentiment_label,
            "स्कोर": sentiment_score
        })
    else:
        st.warning("कृपया पहले कुछ टेक्स्ट लिखें।")
