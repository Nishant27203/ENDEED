"""Recommendation logic powered by Endee vector search."""

from __future__ import annotations

import os
import random
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """Load embedding model once per process for faster responses."""
    return SentenceTransformer(MODEL_NAME)


def get_recommendations(
    movie_title: str,
    top_k: int = 5,
    industry_filter: Optional[str] = None,
    genre_filter: Optional[str] = None,
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
    if genre_filter and genre_filter != "All":
        df = df[df["genre"].str.casefold() == genre_filter.strip().casefold()].reset_index(drop=True)
        if df.empty:
            return {
                "status": "not_found",
                "message": f"No movies found for genre '{genre_filter}'.",
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

    model = _get_embedding_model()
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

    client = get_endee_client()
    index = client.get_index(name=index_name)

    filter_payload = None
    filter_payload = []
    if industry_filter and industry_filter != "All":
        filter_payload.append({"industry": {"$eq": industry_filter.strip().casefold()}})
    if genre_filter and genre_filter != "All":
        filter_payload.append({"genre": {"$eq": genre_filter.strip().casefold()}})
    if not filter_payload:
        filter_payload = None

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


def get_recommendations_by_preferences(
    top_k: int = 5,
    industry_filter: Optional[str] = None,
    genre_filter: Optional[str] = None,
    show_all: bool = False,
    csv_path: str = DATA_PATH,
    index_name: str = INDEX_NAME,
) -> Dict[str, object]:
    """
    Recommend movies from user-selected preferences only (no movie title input).
    """
    df = load_movies(csv_path)
    if industry_filter and industry_filter != "All":
        df = df[df["industry"].str.casefold() == industry_filter.strip().casefold()].reset_index(drop=True)
    if genre_filter and genre_filter != "All":
        df = df[df["genre"].str.casefold() == genre_filter.strip().casefold()].reset_index(drop=True)

    if df.empty:
        return {
            "status": "not_found",
            "message": "No movies match the selected filters. Try different filters.",
            "results": [],
        }

    global _EMBEDDINGS_READY
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

    filter_payload = []
    if industry_filter and industry_filter != "All":
        filter_payload.append({"industry": {"$eq": industry_filter.strip().casefold()}})
    if genre_filter and genre_filter != "All":
        filter_payload.append({"genre": {"$eq": genre_filter.strip().casefold()}})
    if not filter_payload:
        filter_payload = None

    prompt = "Recommend great movies"
    if industry_filter and industry_filter != "All":
        prompt += f" from {industry_filter}"
    if genre_filter and genre_filter != "All":
        prompt += f" in {genre_filter} genre"

    model = _get_embedding_model()
    query_vector = model.encode(prompt, normalize_embeddings=True).tolist()

    client = get_endee_client()
    index = client.get_index(name=index_name)

    result_count = len(df) if show_all else min(top_k, len(df))

    try:
        raw_results = index.query(vector=query_vector, top_k=result_count, filter=filter_payload)
    except requests.exceptions.RequestException:
        return {
            "status": "error",
            "message": "Could not query Endee. Please check Endee service, then retry.",
            "results": [],
        }

    recommendations: List[Dict[str, object]] = []
    for item in raw_results:
        meta = item.get("meta", {})
        recommendations.append(
            {
                "title": meta.get("title", "Unknown title"),
                "description": meta.get("description", "No description available."),
                "industry": meta.get("industry", "Unknown"),
                "genre": meta.get("genre", "Unknown"),
                "similarity": float(item.get("similarity", 0.0)),
            }
        )

    return {
        "status": "success",
        "message": f"Found {len(recommendations)} recommended movies for your preferences.",
        "results": recommendations,
    }


def get_movie_titles(
    industry_filter: Optional[str] = None,
    genre_filter: Optional[str] = None,
    csv_path: str = DATA_PATH,
) -> List[str]:
    """Return all titles, optionally filtered by industry and genre."""
    df = load_movies(csv_path)
    if industry_filter and industry_filter != "All":
        df = df[df["industry"].str.casefold() == industry_filter.strip().casefold()]
    if genre_filter and genre_filter != "All":
        df = df[df["genre"].str.casefold() == genre_filter.strip().casefold()]
    return sorted(df["title"].astype(str).tolist())


def get_all_movie_titles_by_industry(
    industry_filter: Optional[str] = None, csv_path: str = DATA_PATH
) -> List[str]:
    """Backward-compatible wrapper for older UI calls."""
    return get_movie_titles(industry_filter=industry_filter, genre_filter=None, csv_path=csv_path)


def get_available_industries(csv_path: str = DATA_PATH) -> List[str]:
    """Return unique movie industries from dataset."""
    df = load_movies(csv_path)
    industries = sorted(set(df["industry"].astype(str).tolist()))
    return ["All"] + industries


def get_available_genres(csv_path: str = DATA_PATH) -> List[str]:
    """Return unique movie genres from dataset."""
    df = load_movies(csv_path)
    genres = sorted(set(df["genre"].astype(str).tolist()))
    return ["All"] + genres


def get_random_movie(
    industry_filter: Optional[str] = None,
    genre_filter: Optional[str] = None,
    csv_path: str = DATA_PATH,
) -> Optional[str]:
    """Return one random movie title, optionally filtered by industry."""
    titles = get_movie_titles(
        industry_filter=industry_filter,
        genre_filter=genre_filter,
        csv_path=csv_path,
    )
    if not titles:
        return None
    return random.choice(titles)


def get_dataset_summary(csv_path: str = DATA_PATH) -> Dict[str, object]:
    """Return lightweight dataset stats for UI display."""
    df = load_movies(csv_path)
    return {
        "total_movies": int(len(df)),
        "industries": sorted(set(df["industry"].astype(str).tolist())),
        "genres": sorted(set(df["genre"].astype(str).tolist())),
    }


def check_endee_connection() -> Dict[str, str]:
    """Check whether Endee is reachable and return status text for UI."""
    try:
        client = get_endee_client()
        client.list_indexes()
        return {"status": "connected", "message": "Endee status: Connected"}
    except requests.exceptions.RequestException:
        return {
            "status": "disconnected",
            "message": "Endee status: Disconnected (start server on http://127.0.0.1:8080)",
        }
