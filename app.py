
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


# -----------------------------
# Background color styling
# -----------------------------

st.set_page_config(
    page_title="Academic Document QA",
    page_icon="📚",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #e3f2fd,
            #bbdefb,
            #90caf9
        );
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #123b5d !important;
    }

    p, label, .stMarkdown {
        color: #173b57;
    }

    div[data-testid="stTextInput"] input {
        background-color: #ffffff;
        color: #173b57;
        border: 1px solid #90caf9;
        border-radius: 8px;
    }

    div.stButton > button {
        background-color: #1565c0;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #0d47a1;
        color: white;
        border: none;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Tokenization
# -----------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# -----------------------------
# Load document chunks
# -----------------------------

@st.cache_data
def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# -----------------------------
# Load models and indexes
# -----------------------------

@st.cache_resource
def load_models_and_indexes():
    chunks = load_chunks("data/processed/chunks.json")

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

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


# -----------------------------
# Hybrid search: BM25 + FAISS
# -----------------------------

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

    max_bm25 = max(bm25_scores) if len(bm25_scores) else 0

    if max_bm25 > 0:
        normalized_bm25 = [
            score / max_bm25 for score in bm25_scores
        ]
    else:
        normalized_bm25 = [
            0.0 for _ in bm25_scores
        ]

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
        if index >= 0
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
# Application title
# -----------------------------

st.title("📚 Academic Document Question Answering")

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
    placeholder="Example: What is supervised learning?",
    key="question_input"
)


# -----------------------------
# Voice input
# -----------------------------

st.write("Or use your voice:")


def record_voice_question():
    try:
        with st.spinner("Recording... Speak now!"):
            record_audio()

        with st.spinner("Converting speech to text..."):
            recognized_question = speech_to_text()

        if recognized_question and recognized_question.strip():
            st.session_state["question_input"] = (
                recognized_question.strip()
            )
            st.session_state["voice_error"] = ""

        else:
            st.session_state["voice_error"] = (
                "No speech was recognized. Please try again."
            )

    except Exception as error:
        st.session_state["voice_error"] = (
            f"Voice input failed: {error}"
        )


def clear_question():
    st.session_state["question_input"] = ""
    st.session_state["voice_error"] = ""


voice_col, clear_col = st.columns([2, 1])

with voice_col:
    st.button(
        "🎤 Record Question",
        on_click=record_voice_question
    )

with clear_col:
    st.button(
        "Clear Question",
        on_click=clear_question
    )


if st.session_state.get("voice_error"):
    st.warning(st.session_state["voice_error"])


# -----------------------------
# Display recognized question
# -----------------------------

if st.session_state.get("question_input", "").strip():
    st.info(
        f"Current question: {st.session_state['question_input']}"
    )


# -----------------------------
# Ask question
# -----------------------------

if st.button("Ask Question"):

    question = st.session_state.get(
        "question_input", ""
    ).strip()

    if not question:
        st.warning("Please enter or record a question.")

    else:
        try:
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

            if not reranked_results:
                st.warning("No relevant document sections were found.")

            else:
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

        except Exception as error:
            st.error(
                "Unable to generate the answer. "
                "Please try again."
            )
            st.exception(error)
