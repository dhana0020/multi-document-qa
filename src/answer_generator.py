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

Do not mention information that is not supported by the provided context.

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


def get_sources(retrieved_chunks, max_sources=3):
    sorted_chunks = sorted(
        retrieved_chunks,
        key=lambda x: x.get("rerank_score", 0),
        reverse=True
    )

    sources = []

    for chunk in sorted_chunks:
        source = {
            "document": chunk["document"],
            "page": chunk["page"]
        }

        if source not in sources:
            sources.append(source)

        if len(sources) >= max_sources:
            break

    return sources