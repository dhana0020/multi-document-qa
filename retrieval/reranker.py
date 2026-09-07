from sentence_transformers import CrossEncoder


def create_reranker():
    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    return model


def rerank(query, results, reranker, top_k=5):
    pairs = [
        (query, result["text"])
        for result in results
    ]

    scores = reranker.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):
        result = result.copy()
        result["rerank_score"] = float(score)
        reranked_results.append(result)

    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]


if __name__ == "__main__":
    print("Loading reranker model...")

    reranker = create_reranker()

    print("\n--------------------------------")
    print("RERANKER READY")
    print("--------------------------------")