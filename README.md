# AI Movie Recommendation System

**Pick a movie you like and discover others that feel similar—not just ones that happen to repeat the same words in the plot summary.**

This repo is a hands-on data science project: it turns movie text into embeddings, stores them in **Endee** (a vector database), and serves recommendations through a simple web UI. The goal is to show how “similar by meaning” differs from “similar by keywords,” in a stack that resembles what you’d see in real search and recommendation pipelines.

## Project overview

Choosing what to watch next is a fuzzy human problem. We rarely think in pure tags; we think in vibes—same tension, same humor, same kind of world. This app is a small working example of that idea in software: each film is encoded as a dense vector that captures semantic context from its title, description, industry, and genre. When you select a movie, the system asks the vector database for **nearest neighbors** in that space, so suggestions can align with narrative or thematic overlap even when the literal vocabulary differs.

Who is this for? Anyone learning **embeddings**, **vector search**, and **retrieval** without drowning in framework noise—students, portfolio builders, or engineers who want a clear “data in → vectors → index → query → ranked results” story. The UI adds filters (e.g. Hollywood vs. Bollywood, Horror vs. Romance) so you can see how metadata and vector similarity work together in one place.

## Problem statement

Many tutorial recommenders still boil down to **shared words or tags**: if two descriptions don’t overlap much, the model assumes they’re unrelated. That breaks down quickly—synonyms, paraphrases, and “same mood, different wording” all get missed. Worse, users don’t experience recommendations as string overlap; they experience them as “this reminds me of that.”

This project addresses that gap by:

- Representing each movie as a **single semantic embedding** (using `all-MiniLM-L6-v2`) built from structured text fields, not from a bag of isolated keywords.
- Using **cosine similarity search in Endee** so “close in vector space” maps to “close in meaning” for practical top‑k results.
- Keeping the pipeline **transparent**: you can trace from CSV → embedding script → index → query path, which makes it easier to debug, extend, or swap components later.

In short: the problem is *shallow text similarity masquerading as taste*; the approach here is *meaning-aware retrieval backed by a real vector index*.

## What this app does

- Loads a movie dataset with `title`, `description`, `industry`, and `genre`
- Generates embeddings using `all-MiniLM-L6-v2`
- Stores vectors + metadata in Endee
- Recommends similar movies with vector search
- Supports filters by industry/type (Hollywood, Bollywood, Tollywood, Korean, Anime, etc.)
- Supports genre/sections (Horror, Animation, Action, Drama, Crime, Romance, etc.)
- Supports top recommendations and "show all" mode
- Provides a modern dark UI with cards, similarity score, and random movie picker

## How recommendation works (simple flow)

1. Convert each movie into one text block:
   - `title + description + industry + genre`
2. Generate a 384-dim embedding for each movie
3. Store in Endee index (`movies_index`) using cosine similarity
4. User selects a movie and optional industry + genre filter
5. Query Endee with the selected movie embedding
6. Return the most similar movies (excluding the same movie)

## How Endee is used

Endee is the core retrieval engine in this project.

- Index creation: `client.create_index(...)`
- Index access: `client.get_index(...)`
- Upsert vectors: `index.upsert(...)`
- Similarity search: `index.query(...)`

## Project structure

```text
.
├── gradio_app.py           # Main web UI (recommended)
├── app.py                  # Streamlit UI (optional)
├── embeddings.py           # Embedding generation + Endee storage
├── recommend.py            # Recommendation logic + helpers
├── data/
│   └── movies.csv          # Dataset
├── docker-compose.yml      # Local Endee server setup
├── requirements.txt
└── README.md
```

## Setup (quick start)

### 1) Open project

```bash
cd /path/to/ENDEED
```

### 2) Create and activate venv

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Start Endee

```bash
docker compose up -d
```

You can check Endee dashboard at [http://localhost:8080](http://localhost:8080).

## Run the app

### Recommended UI (Gradio)

```bash
python gradio_app.py
```

Then open [http://localhost:7860](http://localhost:7860).

## Live deploy (share with anyone)

If you want people to test your app without downloading code, this is the easiest setup:

### Option 1: Deploy on Render (from GitHub)

1. Push latest code to GitHub (already done).
2. Go to [Render Dashboard](https://dashboard.render.com/) and create:
   - **Web Service** (for this Python app)
   - **Private Service** (for Endee database using Docker image: `endeeio/endee-server:latest`)
3. Configure Web Service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python gradio_app.py`
4. Set environment variable in Web Service:
   - `ENDEE_BASE_URL` = internal URL of Endee service + `/api/v1`
   - Example: `http://<your-endee-service-host>:8080/api/v1`
5. Deploy both services.
6. Share the Render web URL.

### Option 2: Deploy app UI and connect Endee Cloud

If you use Endee Cloud, set:
- `ENDEE_BASE_URL` to your Endee Cloud API URL
- `ENDEE_AUTH_TOKEN` (if required)

Then deploy this repo on any Python host (Render, Railway, etc.) with start command:
`python gradio_app.py`

### Optional UI (Streamlit)

```bash
streamlit run app.py
```

## Environment variables (optional)

- `ENDEE_BASE_URL` (default is local `http://127.0.0.1:8080/api/v1`)
- `ENDEE_AUTH_TOKEN` (if Endee auth is enabled)

## Troubleshooting

- **"Endee server is not running"**  
  Run: `docker compose up -d`

- **No recommendations shown**  
  Refresh app once and try a movie from the dropdown list.

- **First request is slow**  
  Normal on first run: model files are being loaded.

## Notes for submission/demo

This project is intentionally designed to be beginner-friendly but still industry-relevant:
- modular Python files
- semantic search with sentence transformers
- real vector database integration (Endee)
- polished UI and practical filters
