import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Summarizer", layout="wide")

@st.cache_resource
def load_model():
    models = {}
    models['summ'] = pipeline('summarization', model="facebook/bart-large-cnn")
    models['sent'] = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
    return models

# Load models once
models = load_model()

st.title("Summarizer & Sentiment Analyzer!!")
st.write("Paste the text to get summary as well as sentiment report.")

col1, col2 = st.columns([3,1])

with col2:
    max_length = st.slider("Max Summary length:", 30, 400, 120)
    min_length = st.slider("Min Summary length:", 10, 200, 30)

text = col1.text_area("Text to summarize:", height=300, placeholder="Enter text here...")

if st.button('Analyze!!'):
    if not text.strip():
        st.warning("Please enter some text first!!")
    else:
        with st.spinner("Summarizing..."):
            try:
                # Try summarizing whole text
                summary = models['summ'](
                    text,
                    max_length=max_length,
                    min_length=min_length
                )[0]['summary_text']
            except Exception:
                # Fallback: split text into chunks
                sentences = text.split('.')
                chunks = [
                    ".".join(sentences[i:i+8])
                    for i in range(0, len(sentences), 8)
                ]
                parts = [
                    models['summ'](c, max_length=max_length, min_length=min_length)[0]['summary_text']
                    for c in chunks if c.strip()
                ]
                summary = " ".join(parts)

            # Sentiment analysis (limit input size)
            sentiment = models['sent'](text[:1000])

            # Output
            st.subheader("📌 Summary")
            st.write(summary)

            st.subheader("📊 Sentiment")
            st.json(sentiment)
