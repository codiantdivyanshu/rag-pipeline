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
from groq import Groq
from langchain_community.vectorstores import FAISS

# Step 6: Query with Groq API (LLaMA-3)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Search FAISS index for relevant chunks
query = "What is this document about?"
retrieved_docs = vectorstore.similarity_search(query, k=3)
retrieved_texts = [doc.page_content for doc in retrieved_docs]

# Prepare context and prompt
context = "\n\n".join(retrieved_texts)

# Stream Groq LLM response
completion = client.chat.completions.create(
    model="llama3-70b-8192",  # Use correct model name from Groq
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}"},
        {"role": "user", "content": f"Question:\n{query}"}
    ],
    temperature=0.7,
    max_tokens=512,
    top_p=1,
    stream=True
)

# Print streamed response
print("\n\nAnswer:\n")
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")


# Extract text content
texts = [chunk.page_content for chunk in chunks]

# Generate embeddings using local model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, convert_to_numpy=True)

# Save FAISS index
import faiss
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "vector_index.faiss")

# Save texts
import pickle
with open("chunks.pkl", "wb") as f:
    pickle.dump(texts, f)


