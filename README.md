# 📚 Multi-Document QA Chat Bot

An **intelligent conversational AI chatbot** that allows you to upload and query **multiple documents** and have natural conversations about their content.  
Built with **LangChain**, **Groq LLM**, **ChromaDB**, and **Streamlit**.

---

## ✨ Features

- 📄 **Multi-Document Support** – Load and query multiple PDF documents simultaneously
- 💬 **Conversational Memory** – Remembers previous questions and maintains context
- 🔍 **Semantic Search** – Vector embeddings for intelligent document retrieval
- 🤖 **AI-Powered Responses** – Fast, accurate answers using Groq LLM
- 🎨 **Chat Interface** – Clean chat UI with Streamlit + streamlit-chat
- ⚡ **Fast Performance** – Optimized with ChromaDB and efficient chunking
- 🔄 **Session Persistence** – Chat history maintained during the session

---

## 🧠 Use Cases

- Research paper analysis  
- Legal document review  
- Technical documentation Q&A   
- Meeting notes and reports analysis  

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---------|-----------|---------|
| Frontend | Streamlit + streamlit-chat | Web interface |
| LLM | Groq (`openai/gpt-oss-120b`) | Fast inference |
| Embeddings | HuggingFace (sentence-transformers) | Vectorization |
| Vector DB | ChromaDB | Similarity search |
| Framework | LangChain (LCEL) | RAG orchestration |
| PDF Loader | PyPDFLoader | Text extraction |
| Language | Python 3.8+ | Core logic |

---

## 📋 Prerequisites

- Python 3.8+
- Groq API Key (free tier)
- Supported Files: PDFs / DOCX / TXT files to be present in /docs folder.
---

## 🚀 Installation

### Clone the Repository
```bash
git clone https://github.com/Ganesh153/Multi-Document-QA-ChatBot.git
```

### Create Virtual Environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables
Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📦 Dependencies

```txt
streamlit>=1.28.0
streamlit-chat>=0.1.1
langchain>=0.1.0
langchain-community>=0.0.20
langchain-core>=0.1.0
langchain-groq>=0.0.1
langchain-huggingface>=0.0.1
langchain-text-splitters>=0.0.1
python-dotenv>=1.0.0
chromadb>=0.4.18
pypdf>=3.17.0
sentence-transformers>=2.2.0
```

---

## 🎯 Usage

```bash
streamlit run multi_doc_chat.py
```

Open: http://localhost:8501

---

## 📁 Project Structure

```
multi-doc-chatbot/
├── multi_doc_chat.py
├── load_docs.py
├── .env
├── requirements.txt
├── README.md
├── docs/
├── chroma_db/
└── .gitignore
```

---

## 🛑 .gitignore

```gitignore
.env
.venv/
chroma_db/
docs/
__pycache__/
*.pyc
.DS_Store
Thumbs.db
```

---
