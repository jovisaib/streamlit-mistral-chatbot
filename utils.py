import streamlit as st
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

mistral_api_key = st.secrets["MISTRALAI_API_KEY"]


@st.cache_resource
def load_chain():
		llm = ChatMistralAI(mistral_api_key=mistral_api_key)
		
		vector_store = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
		retriever = vector_store.as_retriever(search_kwargs={"k": 3})
		
		memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")
		
		template = """
    You are an AI assistant for answering questions about the Xuberan products.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say 'Sorry, I don't know ... 😔. 
    Don't try to make up an answer.
    If the question is not about Xuberan products, politely inform them that you are tuned to only answer questions about the Xuberan products.
    
    {context}
    Question: {question}
    Helpful Answer:"""
		
		chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                        retriever=retriever, 
                                                        memory=memory, 
                                                        get_chat_history=lambda h : h,
                                                        verbose=True)
		
		QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
		chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
		
		return chain