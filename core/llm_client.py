from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY_QUERY = os.getenv("GROQ_API_KEY")
if not API_KEY_QUERY:
    raise RuntimeError("Missing GROQ_API_KEY in environment. Check .env file.")

client = Groq(api_key=API_KEY_QUERY)

def query_llm_with_context(query, context):
    """
    Send query + context to Groq LLM and return the generated answer.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM query failed: {str(e)}")
