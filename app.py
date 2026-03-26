import os
import streamlit as st
import google.generativeai as genai
from transformers import pipeline

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

model = load_model()

st.title("AI Chatbot")

user_input = st.text_input("Ask something:")

if user_input:
    response = gemini_model.generate_content(user_input)
    st.write("Gemini:", response.text)

    result = model(user_input, max_length=50)
    st.write("Transformer:", result[0]["generated_text"])
