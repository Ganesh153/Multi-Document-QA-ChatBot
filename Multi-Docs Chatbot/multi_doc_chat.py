import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from load_docs import load_docs

load_dotenv(find_dotenv())

# IMPORTANT: Initialize ALL session state variables FIRST
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None

# Initialize LLM and embeddings
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.2
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load and process documents only once
if st.session_state['vectorstore'] is None:
    with st.spinner("Loading documents..."):
        docs = load_docs()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=10
        )
        chunks = text_splitter.split_documents(docs)
        st.session_state['vectorstore'] = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

retriever = st.session_state['vectorstore'].as_retriever(search_kwargs={"k": 5})

template = """Answer the question based on the following context and chat history.
Use the chat history to understand the context of the conversation, but prioritize information from the context below.

Context:
{context}

Chat History:
{chat_history}

Current Question: {question}

Answer in detail:
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(chat_history):
    if not chat_history:
        return "No previous conversation."
    formatted = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)

def get_chat_history():
    return st.session_state.get('chat_history', [])

qa_chain = (
    {
        "context": retriever | format_docs,
        "chat_history": lambda x: format_chat_history(get_chat_history()),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

st.title("📚 Docs QA Bot using LangChain")
st.header("Ask anything about your documents...")

# Display chat history (oldest to newest - FIXED ORDER)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

# Get user input
user_input = st.chat_input("Ask a question regarding your documents...")

if user_input:
    with st.spinner("Thinking..."):
        answer = qa_chain.invoke(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(answer)
        st.session_state['chat_history'].append(HumanMessage(content=user_input))
        st.session_state['chat_history'].append(AIMessage(content=answer))
        
        st.rerun()