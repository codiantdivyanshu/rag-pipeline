🔍 Local RAG (Retrieval-Augmented Generation) Pipeline

This project implements a simple and effective **Retrieval-Augmented Generation (RAG)** system that allows you to upload a PDF, break it into chunks, embed the content, and ask natural language questions — all locally on your machine.

It also includes a FastAPI backend for serving the RAG pipeline via REST API.



📦 Features

 ✅ Load and extract text from any PDF
 ✅ Chunk and embed the document using `sentence-transformers`
 ✅ Store and reuse embeddings locally
 ✅ Query the document with natural language
 ✅ Retrieve top relevant chunks and generate answers using Groq LLM
 ✅ Optional FastAPI server for building a web interface or API integration


🧠 Tech Stack

- Python 3.10+
- PyMuPDF (fitz)
- Sentence Transformers (`all-mpnet-base-v2`)
- Groq API (LLM backend)
- FastAPI + Uvicorn (for API)
- Torch + Numpy + Pandas

📂 Folder Structure





---

🛠️ Installation

1. Clone the repository:

git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline

2.Create and activate a virtual environment  

python -m venv venv
source venv/bin/activate 

3.Install dependencies:

pip install -r requirements.txt

4.Create a .env file and add your Groq API key:

GROQ_API_KEY = 

