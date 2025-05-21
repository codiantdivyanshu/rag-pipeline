# rag_pipeline.py

import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load environment variables from .env file (for OpenAI API key)
load_dotenv()

# Step 1: Load your document(s)
loader = TextLoader("data.txt")  # Make sure this file exists
documents = loader.load()

# Step 2: Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 3: Convert text chunks into embeddings and store in vector DB
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 4: Set up the retriever + LLM chain
retriever = vectorstore.as_retriever()
llm = OpenAI()  # Uses OPENAI_API_KEY from .env
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 5: Ask a question
query = "What is Retrieval-Augmented Generation?"
answer = qa_chain.run(query)

print(f"Q: {query}")
print(f"A: {answer}")
