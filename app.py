"""Streamlit frontend for AI movie recommendations."""

from __future__ import annotations

import streamlit as st

from recommend import get_recommendations

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f1117;
            color: #f5f7fa;
        }
        .main-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
        }
        .movie-card {
            background: linear-gradient(135deg, #1a1d2b 0%, #171a24 100%);
            border: 1px solid #2d3348;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        .movie-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #f9fbff;
            margin-bottom: 0.35rem;
        }
        .movie-description {
            font-size: 0.95rem;
            color: #d8dbe6;
            margin-bottom: 0.5rem;
        }
        .movie-score {
            color: #89b4ff;
            font-size: 0.9rem;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎬 AI Movie Recommendation System</div>', unsafe_allow_html=True)
st.write("Find movies similar to your favorite title using semantic search and Endee vector database.")

movie_input = st.text_input("Enter a movie name", placeholder="e.g., Inception")
clicked = st.button("Get Recommendations", use_container_width=True)

if clicked:
    with st.spinner("Finding similar movies..."):
        response = get_recommendations(movie_input, top_k=5)

    status = response.get("status")
    message = response.get("message", "")
    results = response.get("results", [])

    if status == "success":
        st.success(message)
        for movie in results:
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">{movie['title']}</div>
                    <div class="movie-description">{movie['description']}</div>
                    <div class="movie-score">Similarity score: {movie['similarity']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif status == "not_found":
        st.warning(message)
    else:
        st.error(message or "Something went wrong. Please try again.")
