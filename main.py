import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings 
from dotenv import load_dotenv

load_dotenv()


groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ChatGroq With Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census") 
        st.session_state.docs = st.session_state.loader.load()  
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.success("Vector Store DB is Ready")

if prompt1:
    if "vectors" not in st.session_state:
        st.error("Please embed the documents first by clicking the 'Documents Embedding' button.")
        st.stop()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start
    print("Response time:", response_time)

    if "answer" in response:
        st.write(response["answer"])
    else:
        st.error("No response generated. Check your embeddings or LLM settings.")

    if "context" in response:
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.warning("No similar documents found.")