import streamlit as st
from rag_pipeline import load_rag_pipeline

st.set_page_config(page_title="AI Document Assistant", layout="centered")

st.title("📄 AI Document Assistant")
st.write("Ask questions from your uploaded document using LangChain + RAG")

qa_chain = load_rag_pipeline("data/sample_document.pdf")

query = st.text_input("Enter your question")

if query:
    with st.spinner("Generating answer..."):
        answer = qa_chain.run(query)
        st.success("Answer:")
        st.write(answer)
