import json
import re

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
    Create a BM25 index from document chunks.
    """

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def search_bm25(query, chunks, bm25, top_k=5):
    """
    Search the BM25 index and return the top-k chunks.
    """

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for index in ranked_indices:

        result = chunks[index].copy()

        result["score"] = float(scores[index])

        results.append(result)

    return results


if __name__ == "__main__":

    chunks_file = "data/processed/chunks.json"

    chunks = load_chunks(chunks_file)

    bm25 = create_bm25_index(chunks)

    query = input("\nEnter your question: ")

    results = search_bm25(
        query,
        chunks,
        bm25,
        top_k=5
    )

    print("\n--------------------------------")
    print("BM25 SEARCH RESULTS")
    print("--------------------------------")

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(f"Document: {result['document']}")
        print(f"Page: {result['page']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}")