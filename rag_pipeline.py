

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
\
load_dotenv()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def init_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(query, vectorstore, k=3):
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in retrieved_docs]

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
        stream=False
    )
    
    return completion.choices[0].message["content"]

def rag_pipeline(query, vectorstore, client, k=3):
    context_chunks = retrieve(query, vectorstore, k)
    context = "\n\n".join(context_chunks)
    answer = generate(query, context, client)
    return answer.strip()

if __name__ == "__main__":
    pdf_path = input("Enter path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        exit(1)

    docs = PyPDFLoader(pdf_path).load()
    print(f"Loaded {len(docs)} documents.")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    vectorstore = create_vectorstore(chunks)
    print("FAISS vector store created.")

    groq_client = init_groq_client()

    # Optional: warm-up query
    query = "What are the key points in the document?"
    response = rag_pipeline(query, vectorstore, groq_client)
    print("\nResponse:\n", response)

    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = rag_pipeline(query, vectorstore, groq_client)
        print("\nAnswer:\n", answer)
