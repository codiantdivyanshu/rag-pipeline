# Imports - make sure these are updated to latest
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

# Optional: load environment variables (e.g., OPENAI_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# Step 3: Load and chunk documents
def load_docs(path):
    documents = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, filename))
            documents.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(os.path.join(path, filename))
            documents.extend(loader.load())
    return documents

docs = load_docs("docs")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 4: Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
embeddings = embedding_model


# Step 5: Create a FAISS vector store from chunks
vectorstore = FAISS.from_documents(chunks, embeddings)

# Continue with retrieval, querying, etc.

