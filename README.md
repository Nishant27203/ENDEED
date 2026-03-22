<div align="center">

# AI Movie Recommendation System

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=22&pause=1000&color=38BDF8&center=true&vCenter=true&width=600&lines=Semantic+search+for+movies;Embeddings+%2B+Endee+vector+database;Gradio+%26+Streamlit+UIs)](https://github.com/Nishant27203/ENDEED)

**Pick a film you love—get suggestions that match the *meaning*, not just overlapping keywords.**

[![GitHub stars](https://img.shields.io/github/stars/Nishant27203/ENDEED?style=flat-square&logo=github)](https://github.com/Nishant27203/ENDEED/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Nishant27203/ENDEED?style=flat-square&logo=github)](https://github.com/Nishant27203/ENDEED/network/members)
[![License](https://img.shields.io/github/license/Nishant27203/ENDEED?style=flat-square)](https://github.com/Nishant27203/ENDEED/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/status-active-success?style=flat-square)](https://github.com/Nishant27203/ENDEED)
[![Python](https://img.shields.io/badge/python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

</div>

---

## About the Project

This repo is a hands-on **data science** project: movie text is turned into **embeddings**, stored in **[Endee](https://docs.endee.io/)** (a vector database), and served through a web UI. It shows how *“similar by meaning”* differs from *“similar by keywords”*—the same pattern used in many real **search**, **“more like this”**, and **RAG** pipelines.

Choosing what to watch is a fuzzy, human problem: we think in tone and theme, not raw tags. Here, each film becomes a dense vector from its `title`, `description`, `industry`, and `genre`. You pick a movie (and optional filters); the app queries Endee for **nearest neighbors** in that space—so overlaps in narrative or vibe can surface even when wording differs.

**Problem many demos skip:** keyword-style matching fails on synonyms, paraphrases, and “same mood, different words.” This project uses a **single semantic embedding** per movie (`all-MiniLM-L6-v2`), **cosine similarity** in Endee for top‑k results, and a **traceable** path: CSV → embeddings → index → query.

**Who it’s for:** learners and builders who want a clear **data → vectors → index → ranked results** story—without unnecessary framework noise.

---

## Features

- **Semantic recommendations** — Similarity from embeddings, not just shared words in descriptions  
- **Endee vector store** — Vectors + metadata with fast similarity search  
- **Filters** — Industry (e.g. Hollywood, Bollywood, Tollywood, Korean, Anime) and genre (e.g. Horror, Animation, Action)  
- **Scores & modes** — Similarity scores, top picks, and “show all” style browsing  
- **Modern UI** — Dark-themed cards (Gradio); optional Streamlit UI  
- **Random picker** — Quick way to explore the catalog  
- **Local Endee** — `docker compose` for a reproducible stack  

---

## Tech Stack

| Layer | Tools |
|--------|--------|
| **Language** | Python |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) · `all-MiniLM-L6-v2` (384-dim) |
| **ML runtime** | PyTorch · `torch` / `torchvision` (see `requirements.txt`) |
| **Vector DB** | [Endee](https://docs.endee.io/) — `create_index`, `upsert`, `query` |
| **Web UI** | [Gradio](https://gradio.app/) (primary) · [Streamlit](https://streamlit.io/) (`app.py`, optional) |
| **Data** | pandas · `data/movies.csv` |
| **Ops** | Docker Compose for local Endee |

---

## How Recommendation Works

1. Build one text block per movie: `title + description + industry + genre`  
2. Embed with `all-MiniLM-L6-v2` → **384-dimensional** vectors  
3. Store in Endee index **`movies_index`** (cosine similarity)  
4. User selects a movie and optional **industry** + **genre** filters  
5. Query Endee with that movie’s embedding  
6. Return the most similar titles (**excluding** the query movie)  

### Endee in this project

Endee is the retrieval engine:

- Index: `client.create_index(...)`  
- Access: `client.get_index(...)`  
- Write: `index.upsert(...)`  
- Search: `index.query(...)`  

---

## Project Structure

```text
.
├── gradio_app.py           # Main web UI (recommended)
├── app.py                  # Streamlit UI (optional)
├── embeddings.py           # Embedding generation + Endee storage
├── recommend.py            # Recommendation logic + helpers
├── data/
│   └── movies.csv          # Dataset
├── docker-compose.yml      # Local Endee server
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### Prerequisites

- Python 3.x  
- [Docker](https://www.docker.com/) (for local Endee)  

### 1. Clone and enter the project

```bash
git clone https://github.com/Nishant27203/ENDEED.git
cd ENDEED
```

### 2. Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Endee

```bash
docker compose up -d
```

Dashboard: [http://localhost:8080](http://localhost:8080)

### 5. Run the app (Gradio — recommended)

```bash
python gradio_app.py
```

Open [http://localhost:7860](http://localhost:7860).

### Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `ENDEE_BASE_URL` | API base (default: `http://127.0.0.1:8080/api/v1`) |
| `ENDEE_AUTH_TOKEN` | If your Endee instance requires auth |

### Optional: Streamlit UI

Install Streamlit if you use `app.py`:

```bash
pip install streamlit
streamlit run app.py
```

---

## Deployment

Share the app without asking everyone to clone the repo.

### Option A — Render (app + Endee)

1. Push this repo to GitHub.  
2. In [Render](https://dashboard.render.com/): create a **Web Service** (this app) and a **Private Service** for Endee (`endeeio/endee-server:latest`).  
3. Web service: **Build** `pip install -r requirements.txt` · **Start** `python gradio_app.py`  
4. Set **`ENDEE_BASE_URL`** to your Endee service URL + `/api/v1` (e.g. `http://<endee-host>:8080/api/v1`).  
5. Deploy and share the web URL.  

### Option B — Any host + Endee Cloud

Point **`ENDEE_BASE_URL`** (and **`ENDEE_AUTH_TOKEN`** if needed) at your cloud Endee API, then deploy with start command `python gradio_app.py` (e.g. Render, Railway).

---

## Screenshots / Preview

> **Placeholder** — Replace with a GIF, short screen recording, or static screenshots of the Gradio UI when you have them.

| Preview | What to show |
|---------|----------------|
| ![UI placeholder](https://via.placeholder.com/720x405/0f172a/38bdf8?text=Gradio+movie+recommender+UI) | Dark theme, movie cards, similarity scores, industry/genre filters |

**GIF / demo:** Add a file under `docs/` (e.g. `docs/demo.gif`) and reference it here: `![Demo](./docs/demo.gif)`.

---

## Troubleshooting

| Issue | What to try |
|--------|----------------|
| “Endee server is not running” | `docker compose up -d` |
| No recommendations | Refresh once; pick a movie from the dropdown |
| First request is slow | Normal while the embedding model downloads/loads |

---

## Notes (submission / demo)

This project is intentionally **beginner-friendly** but **industry-shaped**:

- Modular Python layout  
- Semantic search with sentence transformers  
- Real vector database integration (Endee)  
- Polished UI with practical filters  

---

## Conclusion

This is a focused slice of **content-similarity** systems: text → vectors → **nearest-neighbor** search → UI. It is **not** a full production recommender (no collaborative filtering at scale, no A/B harness, no live feedback loop)—but it **does** show the embedding + retrieval pattern behind many modern discovery features.

**Takeaway:** matching **meaning** is often more useful than matching **strings**. Next steps could include richer metadata, hybrid keyword + vector search, re-ranking, or reusing the same Endee index from another client.

---

## Contributing

Contributions are welcome.

1. Fork the repository  
2. Create a branch (`git checkout -b feature/your-idea`)  
3. Commit your changes (`git commit -m 'Add concise description'`)  
4. Push to the branch (`git push origin feature/your-idea`)  
5. Open a Pull Request  

Please keep changes focused and consistent with the existing code style.

---

## License

This repository may not include a `LICENSE` file yet. [Add a license](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) (e.g. MIT) to clarify how others may use the code. Until then, all rights are reserved by default—check with the author if unsure.

---

## Author / Contact

**Nishant** · [@Nishant27203](https://github.com/Nishant27203)

[![GitHub](https://img.shields.io/badge/GitHub-Nishant27203-181717?style=flat-square&logo=github)](https://github.com/Nishant27203)

<!-- Optional: add your LinkedIn profile URL -->
<!-- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/YOUR_USERNAME/) -->

---

## Show Your Support

If this project helped you learn about embeddings or vector search, consider **starring the repo** on GitHub—it helps others discover it and keeps motivation high for future updates.

[![Star on GitHub](https://img.shields.io/github/stars/Nishant27203/ENDEED?style=social)](https://github.com/Nishant27203/ENDEED)

---

<div align="center">

<sub>Built with semantic search, Endee, and curiosity.</sub>

</div>
