
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=api_key)


# Try the primary model first, then a fallback model.
# Confirm these model IDs are available to your API key.
MODELS = [
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]


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

    last_error = None

    for model_name in MODELS:

        # Retry temporary server errors up to 3 times.
        for attempt in range(3):

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if response.text:
                    return response.text

                last_error = RuntimeError(
                    f"{model_name} returned an empty response."
                )
                break

            except errors.ServerError as error:
                last_error = error

                # Retry only temporary server errors.
                if getattr(error, "code", None) not in (500, 502, 503, 504):
                    raise

                if attempt < 2:
                    time.sleep(2 ** attempt)

            except errors.APIError as error:
                last_error = error

                # Don't retry authentication, invalid request,
                # or other non-server API errors.
                raise

        # If the primary model failed, try the next model.

    raise RuntimeError(
        "Gemini could not generate an answer after retries. "
        "Please try again later or check model availability. "
        f"Last error: {last_error}"
    )


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
