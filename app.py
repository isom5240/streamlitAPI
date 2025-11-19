
import os
import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("Image-to-Text and Text-to-Speech App")
HF_TOKEN = os.environ["HF_TOKEN"]
image_to_text = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning",
    token=HF_TOKEN)
text_to_speech = pipeline(
    "text-to-speech",
    model="facebook/mms-tts-eng",
    token=HF_TOKEN)

uploaded_file = st.file_uploader("Upload an image", 
                                 type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    caption = image_to_text(image)[0]["generated_text"]
    st.write("Caption:", caption)

    audio = text_to_speech(caption)
    audio_path = "speech.wav"
    with open(audio_path, "wb") as f:
        f.write(audio["audio"])

    st.audio(audio_path)
