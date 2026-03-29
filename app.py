import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Updated imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# =========================
# 🔐 API KEY
# =========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found")
    st.stop()

genai.configure(api_key=api_key)

# =========================
# 📄 PDF PATH
# =========================
PDF_PATH = "sample.pdf"

# =========================
# 📥 READ PDF
# =========================
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# =========================
# ✂️ TEXT SPLIT
# =========================
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)

# =========================
# 🧠 VECTOR STORE
# =========================
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index")

# =========================
# 🤖 ANSWER FUNCTION
# =========================
def get_answer(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question)

    context = "\n".join([doc.page_content for doc in docs])

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3
    )

    prompt = f"""
    Answer the question using the context below.
    If answer is not available, say "Not found in document".

    Context:
    {context}

    Question:
    {question}
    """

    response = model.invoke(prompt)
    return response.content

# =========================
# 🎨 UI
# =========================
st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")

st.title("📄 AI PDF Chatbot")
st.write("Ask questions from your PDF")

# =========================
# CREATE VECTOR DB ONCE
# =========================
if not os.path.exists("faiss_index"):
    with st.spinner("Processing PDF..."):
        text = get_pdf_text(PDF_PATH)
        chunks = get_text_chunks(text)
        create_vector_store(chunks)
    st.success("PDF Ready!")

# =========================
# CHAT HISTORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# CHAT INPUT
# =========================
if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
