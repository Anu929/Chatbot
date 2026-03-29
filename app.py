import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Updated LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# =========================
# 🔐 API KEY SETUP
# =========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found!")
    st.stop()

genai.configure(api_key=api_key)

# =========================
# 📄 PDF PATH
# =========================
PDF_PATH = "sample.pdf"   # put your PDF in same folder

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
# ✂️ TEXT SPLITTING
# =========================
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)

# =========================
# 🧠 VECTOR STORE
# =========================
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# =========================
# 🤖 QA CHAIN
# =========================
def get_qa_chain(vector_store):
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_store.as_retriever(),
        chain_type="stuff"
    )

    return qa_chain

# =========================
# 💬 USER INPUT HANDLER
# =========================
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    chain = get_qa_chain(db)

    response = chain.run(user_question)
    return response

# =========================
# 🎨 STREAMLIT UI
# =========================
st.set_page_config(page_title="AI PDF Chatbot", page_icon="🤖")

st.title("📄 AI PDF Chatbot (Gemini + LangChain)")
st.write("Ask questions from your PDF file")

# =========================
# ⚡ CREATE VECTOR DB ONCE
# =========================
if not os.path.exists("faiss_index"):
    with st.spinner("Processing PDF..."):
        raw_text = get_pdf_text(PDF_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
    st.success("PDF processed successfully!")

# =========================
# 💾 CHAT HISTORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# 💬 CHAT INPUT
# =========================
if prompt := st.chat_input("Ask something from the PDF..."):

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = user_input(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
