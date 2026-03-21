# 🎬 AI Movie Recommendation System

A beginner-friendly, end-to-end movie recommendation web app built with Python, Streamlit, Sentence Transformers, and **Endee vector database**.

## Project Overview

This project recommends movies similar to a user-selected movie title by:

1. Converting movie text (title + description) into dense vectors (embeddings)
2. Storing those vectors in Endee
3. Running vector similarity search to return the top matches

The app uses a clean dark UI and is suitable for a data science project/demo.

## Problem Statement

Traditional keyword search misses semantic meaning. For example, two movies may be similar in theme even when they do not share exact words.  
This project solves that by using embedding-based semantic similarity instead of plain text matching.

## How Recommendation Works

1. Load `data/movies.csv` with `title`, `description`, `industry`, and `genre`
2. Build text as: `"{title}. {description} Industry: {industry}. Genre: {genre}."`
3. Generate embeddings using `all-MiniLM-L6-v2` (384 dimensions)
4. Create an Endee index (`movies_index`) with cosine similarity
5. Upsert each movie vector with metadata (`title`, `description`, `industry`, `genre`) and filters
6. On user query:
   - Validate movie title exists in dataset
   - Encode that movie text into a query vector
   - Query Endee for top similar vectors
   - Filter by selected industry/type (Hollywood, Bollywood, Tollywood, Korean, Anime, etc.)
   - Exclude the same movie and show top 5 or all recommendations

## Movie Types Added

The sample dataset now includes multiple movie industries/types:

- Hollywood
- Bollywood
- Tollywood
- Korean
- Anime

## How Endee Is Used as Vector Database

Endee is used in the project for all vector operations:

- Create index: `client.create_index(...)`
- Get index: `client.get_index(...)`
- Store vectors: `index.upsert([...])`
- Similarity search: `index.query(vector=..., top_k=...)`

This makes Endee the core retrieval engine of the recommendation system.

## Project Structure

```text
.
├── app.py                  # Streamlit UI
├── embeddings.py           # Embedding generation + Endee storage
├── recommend.py            # Query + recommendation logic
├── data/
│   └── movies.csv          # Sample movie dataset
├── .streamlit/
│   └── config.toml         # Dark theme settings
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1) Clone or open project

```bash
cd /path/to/project
```

### 2) Create virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Start Endee locally

Follow the official Endee quick start:
- [Endee Quick Start](https://docs.endee.io/quick-start)

By default, this project connects to local Endee (`localhost:8080`).

Optional environment variables:

- `ENDEE_BASE_URL` (example: `http://0.0.0.0:8081/api/v1`)
- `ENDEE_AUTH_TOKEN` (if auth is enabled)

## How To Run the App

### Option A (Recommended): Gradio UI (non-Streamlit)

```bash
python gradio_app.py
```

Open:
- `http://localhost:7860`

### Option B: Streamlit UI

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal.

## Notes

- If a movie title is not found, the app shows a friendly warning.
- The first run may take longer because the embedding model is downloaded.
- Re-running keeps vectors up to date by upserting with stable IDs.
