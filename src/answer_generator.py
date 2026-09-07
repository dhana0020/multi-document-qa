import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=api_key)


def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(
        [
            f"Document: {chunk['document']}\n"
            f"Page: {chunk['page']}\n"
            f"Content: {chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are a question-answering assistant.

Answer the user's question using ONLY the provided document context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided documents."

Give a clear and concise answer.

Question:
{question}

Document Context:
{context}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text