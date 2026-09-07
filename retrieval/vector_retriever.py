import json
import re

import faiss
from sentence_transformers import SentenceTransformer


def tokenize(text):
    """
    Convert text into lowercase word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks(file_path):
    """
    Load processed chunks from JSON.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_vector_index(chunks, model):
    """
    Convert chunks into embeddings and create a FAISS index.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Normalize embeddings so inner product becomes
    # cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_vector(query, chunks, model, index, top_k=5):
    """
    Search the vector index using semantic similarity.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):

        result = chunks[index_position].copy()

        result["score"] = float(score)

        results.append(result)

    return results


if __name__ == "__main__":

    chunks_file = "data/processed/chunks.json"

    print("Loading chunks...")

    chunks = load_chunks(chunks_file)

    print(f"Total chunks: {len(chunks)}")

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Creating vector index...")

    index = create_vector_index(
        chunks,
        model
    )

    print("\n--------------------------------")
    print("VECTOR SEARCH READY")
    print("--------------------------------")

    query = input("\nEnter your question: ")

    results = search_vector(
        query,
        chunks,
        model,
        index,
        top_k=5
    )

    print("\n--------------------------------")
    print("VECTOR SEARCH RESULTS")
    print("--------------------------------")

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(f"Document: {result['document']}")
        print(f"Page: {result['page']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}")