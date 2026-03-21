"""Generate and store movie embeddings in Endee vector database."""

from __future__ import annotations

import os
from typing import List

import pandas as pd
from endee import Endee, Precision
from endee.exceptions import ConflictException

# Prevent transformers from importing TensorFlow/Keras in mixed environments.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_NAME = "movies_index"
EMBEDDING_DIM = 384
DATA_PATH = "data/movies.csv"


def get_endee_client() -> Endee:
    """Create an Endee client, optionally using env vars."""
    token = os.getenv("ENDEE_AUTH_TOKEN")
    base_url = os.getenv("ENDEE_BASE_URL")

    client = Endee(token) if token else Endee()
    if base_url:
        client.set_base_url(base_url)
    return client


def _index_exists(client: Endee, index_name: str) -> bool:
    """Check whether an index already exists."""
    indexes = client.list_indexes()

    # Keep parsing tolerant because SDK response shapes can vary by version.
    candidate_items = indexes
    if isinstance(indexes, dict) and isinstance(indexes.get("indexes"), list):
        candidate_items = indexes["indexes"]

    if isinstance(candidate_items, list):
        for item in candidate_items:
            if isinstance(item, str) and item == index_name:
                return True
            if isinstance(item, dict) and item.get("name") == index_name:
                return True
    return False


def load_movies(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """Load movie dataset and validate expected columns."""
    df = pd.read_csv(csv_path)
    required_cols = {"title", "description", "industry", "genre"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")
    df["industry"] = df["industry"].fillna("Unknown").astype(str)
    df["genre"] = df["genre"].fillna("Unknown").astype(str)
    return df


def build_movie_texts(df: pd.DataFrame) -> List[str]:
    """Build text to embed for each movie."""
    return [
        f"{row.title}. {row.description} Industry: {row.industry}. Genre: {row.genre}."
        for row in df.itertuples(index=False)
    ]


def generate_and_store_embeddings(
    csv_path: str = DATA_PATH,
    index_name: str = INDEX_NAME,
) -> None:
    """
    Generate embeddings for movies and upsert them to Endee.
    Safe to run multiple times; it updates existing vectors by ID.
    """
    df = load_movies(csv_path)
    texts = build_movie_texts(df)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True)

    client = get_endee_client()
    if not _index_exists(client, index_name):
        try:
            client.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
        except ConflictException:
            # Another request/process may create the index simultaneously.
            pass

    index = client.get_index(name=index_name)

    payload = []
    for i, row in enumerate(df.itertuples(index=False)):
        payload.append(
            {
                "id": str(i),
                "vector": embeddings[i].tolist(),
                "meta": {
                    "title": row.title,
                    "description": row.description,
                    "industry": row.industry,
                    "genre": row.genre,
                },
                "filter": {
                    "industry": str(row.industry).strip().casefold(),
                    "genre": str(row.genre).strip().casefold(),
                },
            }
        )

    index.upsert(payload)


if __name__ == "__main__":
    generate_and_store_embeddings()
    print("Movie embeddings generated and stored in Endee.")
