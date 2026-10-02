import streamlit as st
import json
import re
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from retrieval.reranker import create_reranker, rerank

import sys
sys.path.append("src")

from answer_generator import generate_answer, get_sources
from voice_input import record_audio, speech_to_text


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


@st.cache_data
def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_models_and_indexes():
    chunks = load_chunks("data/processed/chunks.json")

    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    vector_index = faiss.IndexFlatIP(dimension)
    vector_index.add(embeddings)

    reranker = create_reranker()

    return chunks, bm25, model, vector_index, reranker


def hybrid_search(
    query,
    chunks,
    bm25,
    vector_index,
    model,
    top_k=10,
    bm25_weight=0.5,
    vector_weight=0.5
):
    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    max_bm25 = max(bm25_scores)

    if max_bm25 > 0:
        normalized_bm25 = [
            score / max_bm25 for score in bm25_scores
        ]
    else:
        normalized_bm25 = [0.0 for _ in bm25_scores]

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

    combined_results = []

    for index in range(len(chunks)):

        bm25_score = normalized_bm25[index]
        vector_score = vector_score_map.get(index, 0.0)

        hybrid_score = (
            bm25_weight * bm25_score
            + vector_weight * vector_score
        )

        result = chunks[index].copy()

        result["bm25_score"] = float(bm25_score)
        result["vector_score"] = float(vector_score)
        result["hybrid_score"] = float(hybrid_score)

        combined_results.append(result)

    combined_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return combined_results[:top_k]


# -----------------------------
# Streamlit interface
# -----------------------------

st.set_page_config(
    page_title="Academic Document QA",
    page_icon="📚",
    layout="wide"
)

st.title("Academic Document Question Answering")

st.write(
    "Ask questions based on the provided academic PDF documents."
)


# -----------------------------
# Load retrieval system
# -----------------------------

with st.spinner("Loading document retrieval system..."):

    chunks, bm25, model, vector_index, reranker = (
        load_models_and_indexes()
    )

st.success("Document retrieval system ready.")


# -----------------------------
# Question input
# -----------------------------

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is supervised learning?"
)


# -----------------------------
# Voice input
# -----------------------------

st.write("Or use your voice:")

if st.button("🎤 Record Question"):

    with st.spinner("Recording... Speak now!"):

        record_audio()

    with st.spinner("Converting speech to text..."):

        question = speech_to_text()

    st.session_state["voice_question"] = question


# Display recognized question
if "voice_question" in st.session_state:

    question = st.session_state["voice_question"]

    st.info(
        f"Recognized question: {question}"
    )


# -----------------------------
# Ask question
# -----------------------------

if st.button("Ask Question"):

    if not question.strip():

        st.warning("Please enter or record a question.")

    else:

        with st.spinner("Searching documents..."):

            results = hybrid_search(
                question,
                chunks,
                bm25,
                vector_index,
                model,
                top_k=10
            )

            reranked_results = rerank(
                question,
                results,
                reranker,
                top_k=5
            )

        with st.spinner("Generating answer..."):

            answer = generate_answer(
                question,
                reranked_results
            )

            sources = get_sources(
                reranked_results
            )

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Sources")

        for source in sources:

            st.write(
                f"- {source['document']} "
                f"(Page {source['page']})"
            )