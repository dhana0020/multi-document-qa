import json
import re

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


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


def create_bm25_index(chunks):
    """
    Create BM25 index.
    """
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def create_vector_index(chunks, model):
    """
    Create FAISS vector index.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def hybrid_search(
    query,
    chunks,
    bm25,
    vector_index,
    model,
    top_k=5,
    bm25_weight=0.5,
    vector_weight=0.5
):
    """
    Combine BM25 and vector search scores.
    """

    # -----------------------------
    # BM25 SEARCH
    # -----------------------------

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    # Normalize BM25 scores
    max_bm25 = max(bm25_scores)

    if max_bm25 > 0:
        normalized_bm25 = [
            score / max_bm25
            for score in bm25_scores
        ]
    else:
        normalized_bm25 = [
            0.0 for _ in bm25_scores
        ]

    # -----------------------------
    # VECTOR SEARCH
    # -----------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    vector_scores, vector_indices = vector_index.search(
        query_embedding,
        len(chunks)
    )

    vector_score_map = {
        int(index): float(score)
        for index, score in zip(
            vector_indices[0],
            vector_scores[0]
        )
    }

    # -----------------------------
    # COMBINE SCORES
    # -----------------------------

    combined_results = []

    for index in range(len(chunks)):

        bm25_score = normalized_bm25[index]

        vector_score = vector_score_map.get(
            index,
            0.0
        )

        hybrid_score = (
            bm25_weight * bm25_score
            +
            vector_weight * vector_score
        )

        result = chunks[index].copy()

        result["bm25_score"] = float(bm25_score)
        result["vector_score"] = float(vector_score)
        result["hybrid_score"] = float(hybrid_score)

        combined_results.append(result)

    # Sort by hybrid score

    combined_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return combined_results[:top_k]


if __name__ == "__main__":

    chunks_file = "data/processed/chunks.json"

    print("Loading chunks...")

    chunks = load_chunks(chunks_file)

    print(f"Total chunks: {len(chunks)}")

    # -----------------------------
    # BM25
    # -----------------------------

    print("\nCreating BM25 index...")

    bm25 = create_bm25_index(chunks)

    # -----------------------------
    # Vector model
    # -----------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Vector index
    # -----------------------------

    print("\nCreating vector index...")

    vector_index = create_vector_index(
        chunks,
        model
    )

    print("\n--------------------------------")
    print("HYBRID SEARCH READY")
    print("--------------------------------")

    query = input("\nEnter your question: ")

    results = hybrid_search(
        query,
        chunks,
        bm25,
        vector_index,
        model,
        top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )

    print("\n--------------------------------")
    print("HYBRID SEARCH RESULTS")
    print("--------------------------------")

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(f"Document: {result['document']}")
        print(f"Page: {result['page']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"BM25 Score: {result['bm25_score']:.4f}")
        print(f"Vector Score: {result['vector_score']:.4f}")
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"Text: {result['text'][:500]}")