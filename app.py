
import streamlit as st
from transformers import pipeline
from io import BytesIO
import os

# Load the TTS model
text_to_speech = pipeline(
    "text-to-speech",
    model="facebook/mms-tts-eng")

# Generate audio
audio = text_to_speech("This is a test")

# audio["audio"] is raw WAV bytes → wrap in BytesIO
audio_bytes = BytesIO(audio["audio"])

# Streamlit can play from file-like objects
st.audio(audio_bytes, format="audio/wav")
