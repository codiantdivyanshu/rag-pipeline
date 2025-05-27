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
rag-pipeline/
│
├── docs/                      # Documentation or design assets
│   └── file.pdf               # Example doc (add diagrams, notes, etc.)
│
├── data.txt                   # Sample input or notes (can be removed if unused)
├── .env                       # Environment variables (e.g., GROQ_API_KEY)
├── .gitignore                 # Git ignore rules
├── LICENSE                    # License file (e.g., MIT)
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── rag_pipeline.py            # Terminal-based pipeline script
├── rag_api.py                 # Shared logic used by both CLI and FastAPI
├── fast_api_app.py            # FastAPI server entry point
│
├── venv/                      # Python virtual environment (excluded from Git)



🛠️ Installation

1. Clone the repository:

git clone https://github.com/codiantdivyanshu/rag-pipeline.git
cd rag-pipeline

2.Create and activate a virtual environment  

python -m venv venv
source venv/bin/activate 

3.Install dependencies:

pip install -r requirements.txt

4.Create a .env file and add your Groq API key:

GROQ_API_KEY = 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

 
:) Run Locally (Terminal)

   python rag_pipeline.py 
   
 :) Run via FastAPI

   uvicorn rag_api:app --reload

   Visit: http://127.0.0.1:8000/docs for interactive Swagger UI.

:) Endpoints

POST /upload-pdf/ — Upload a PDF and generate embeddings

POST /query/ — Submit a natural language question and receive an answer

:) Dependencies 

numpy
torch
pandas
tqdm
requests
PyMuPDF
sentence-transformers
textwrap3
groq
fastapi
uvicorn
python-dotenv

:) License
 
 MIT License

