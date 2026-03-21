# AI Movie Recommendation System

An AI-powered movie recommendation web app built as a practical data science project.
It uses sentence embeddings for semantic understanding and Endee as the vector database for fast similarity search.

## Why this project?

Most basic recommendation demos rely on keyword matching. That works only when titles/descriptions share similar words.
This project goes one step further by using semantic embeddings, so recommendations are based on meaning, not just exact text.

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
