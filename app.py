import streamlit as st
from transformers import pipeline

sentiment_pipeline = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

st.title("Sentiment Analysis with HuggingFace Spaces (ISOM5240)")
st.write("Enter a sentence to analyze its sentiment:")


