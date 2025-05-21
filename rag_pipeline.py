# rag_pipeline.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from groq import Groq

# Load environment variables (e.g., API keys)
load_dotenv()

# Step 1: Load documents from a folder
def load_documents(path):
    docs = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(full_path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(full_path).load())
    return docs

# Step 2: Split documents into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Step 3: Generate embeddings and build FAISS vectorstore
def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Step 4: Initialize Groq LLM client
def init_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

# Step 5: Query vectorstore and send to Groq LLM
def ask_groq(query, vectorstore, client, k=3):
    # Retrieve top k similar chunks
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Construct prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}"},
        {"role": "user", "content": f"Question:\n{query}"}
    ]
    
    # Call LLaMA-3 via Groq
    print("\nAnswer:\n")
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        stream=True
    )
    
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")

# Main logic
if __name__ == "__main__":
    docs = load_documents("docs")
    print(f"Loaded {len(docs)} documents.")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    vectorstore = create_vectorstore(chunks)
    print("FAISS vector store created.")

    groq_client = init_groq_client()

    # Example user query
    query = input("\nEnter your question: ")
    ask_groq(query, vectorstore, groq_client)
