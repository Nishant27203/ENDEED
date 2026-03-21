"""Recommendation logic powered by Endee vector search."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

# Prevent transformers from importing TensorFlow/Keras in mixed environments.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import requests
from sentence_transformers import SentenceTransformer

from embeddings import (
    DATA_PATH,
    INDEX_NAME,
    MODEL_NAME,
    build_movie_texts,
    generate_and_store_embeddings,
    get_endee_client,
    load_movies,
)

_EMBEDDINGS_READY = False


def _normalize_title(title: str) -> str:
    return title.strip().casefold()


def get_recommendations(
    movie_title: str,
    top_k: int = 5,
    industry_filter: Optional[str] = None,
    show_all: bool = False,
    csv_path: str = DATA_PATH,
    index_name: str = INDEX_NAME,
) -> Dict[str, object]:
    """
    Return top similar movies for a given movie title.
    Handles unknown movie titles gracefully.
    """
    if not movie_title or not movie_title.strip():
        return {"status": "error", "message": "Please enter a movie title.", "results": []}

    df = load_movies(csv_path)
    if industry_filter and industry_filter != "All":
        df = df[df["industry"].str.casefold() == industry_filter.strip().casefold()].reset_index(drop=True)
        if df.empty:
            return {
                "status": "not_found",
                "message": f"No movies found for industry '{industry_filter}'.",
                "results": [],
            }

    titles = df["title"].tolist()
    normalized_map = {_normalize_title(t): idx for idx, t in enumerate(titles)}

    query_key = _normalize_title(movie_title)
    if query_key not in normalized_map:
        return {
            "status": "not_found",
            "message": (
                f"'{movie_title}' was not found in the dataset. "
                "Try one of the listed sample movies."
            ),
            "results": [],
        }

    global _EMBEDDINGS_READY
    # Prepare vectors once per app process to keep requests fast/stable.
    try:
        if not _EMBEDDINGS_READY:
            generate_and_store_embeddings(csv_path=csv_path, index_name=index_name)
            _EMBEDDINGS_READY = True
    except requests.exceptions.RequestException:
        return {
            "status": "error",
            "message": (
                "Endee server is not running. Start Endee first, then try again. "
                "Default endpoint: http://127.0.0.1:8080"
            ),
            "results": [],
        }

    query_idx = normalized_map[query_key]
    query_title = titles[query_idx]
    query_text = build_movie_texts(df)[query_idx]

    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

    client = get_endee_client()
    index = client.get_index(name=index_name)

    filter_payload = None
    if industry_filter and industry_filter != "All":
        filter_payload = [{"industry": {"$eq": industry_filter.strip().casefold()}}]

    result_count = len(df) - 1 if show_all else top_k
    result_count = max(1, result_count)

    # Ask for one extra result because the closest item is usually the same movie.
    try:
        raw_results = index.query(vector=query_vector, top_k=result_count + 1, filter=filter_payload)
    except requests.exceptions.RequestException:
        return {
            "status": "error",
            "message": (
                "Could not query Endee. Please check Endee service and ENDEE_BASE_URL, "
                "then retry."
            ),
            "results": [],
        }

    recommendations: List[Dict[str, object]] = []
    for item in raw_results:
        meta = item.get("meta", {})
        candidate_title = meta.get("title", "Unknown title")
        if _normalize_title(candidate_title) == query_key:
            continue

        recommendations.append(
            {
                "title": candidate_title,
                "description": meta.get("description", "No description available."),
                "industry": meta.get("industry", "Unknown"),
                "genre": meta.get("genre", "Unknown"),
                "similarity": float(item.get("similarity", 0.0)),
            }
        )
        if len(recommendations) >= result_count:
            break

    return {
        "status": "success",
        "message": f"Found {len(recommendations)} recommendations for '{query_title}'.",
        "results": recommendations,
    }


def get_all_movie_titles_by_industry(
    industry_filter: Optional[str] = None, csv_path: str = DATA_PATH
) -> List[str]:
    """Return all titles, optionally filtered by industry."""
    df = load_movies(csv_path)
    if industry_filter and industry_filter != "All":
        df = df[df["industry"].str.casefold() == industry_filter.strip().casefold()]
    return sorted(df["title"].tolist())


def get_available_industries(csv_path: str = DATA_PATH) -> List[str]:
    """Return unique movie industries from dataset."""
    df = load_movies(csv_path)
    industries = sorted(set(df["industry"].astype(str).tolist()))
    return ["All"] + industries
