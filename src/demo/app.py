import streamlit as st
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.query_rag import main as rag_main

st.title("RAG QA Tiếng Việt - Biomedical")
st.write("Nhập câu hỏi tiếng Việt để hệ thống trả lời dựa trên KB.")

question = st.text_input("Câu hỏi")

if st.button("Trả lời") and question:
    # Gợi ý: chạy CLI trước khi demo để build vectorstore
    st.info("Hãy chạy build_kb.py trước khi demo")
    st.write("Câu hỏi:", question)
