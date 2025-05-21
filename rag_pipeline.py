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

# Step 5: Retrieve top k similar chunks from vectorstore
def retrieve(query, vectorstore, k=3):
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in retrieved_docs]

# Step 6: Generate answer using Groq LLM with context + query
def generate(query, context, client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}"},
        {"role": "user", "content": f"Question:\n{query}"}
    ]
    
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        stream=False  # Use False here to collect full answer at once
    )
    
    return completion.choices[0].message["content"]

# Step 7: Full RAG pipeline: retrieve + generate answer
def rag_pipeline(query, vectorstore, client, k=3):
    context_chunks = retrieve(query, vectorstore, k)
    context = "\n\n".join(context_chunks)
    answer = generate(query, context, client)
    return answer.strip()

# Main logic
if __name__ == "__main__":
    docs = load_documents("docs")
    print(f"Loaded {len(docs)} documents.")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    vectorstore = create_vectorstore(chunks)
    print("FAISS vector store created.")

    groq_client = init_groq_client()

    # Example user query loop
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = rag_pipeline(query, vectorstore, groq_client)
        print("\nAnswer:\n", answer)
