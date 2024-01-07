import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


loader = NotionDirectoryLoader("notion_content")
documents = loader.load()

markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\\n\\n","\\n","."],
    chunk_size=1500,
    chunk_overlap=100)
docs = markdown_splitter.split_documents(documents)

db = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
db.save_local("faiss_index")

print('Local FAISS index has been successfully saved.')