"""Gradio frontend for AI movie recommendations (non-Streamlit option)."""

from __future__ import annotations

import html
from typing import List, Dict

import gradio as gr

from recommend import (
    check_endee_connection,
    get_available_genres,
    get_available_industries,
    get_dataset_summary,
    get_recommendations_by_preferences,
)


def _result_card(item: Dict[str, object]) -> str:
    title = html.escape(str(item.get("title", "Unknown Title")))
    desc = html.escape(str(item.get("description", "No description available.")))
    industry = html.escape(str(item.get("industry", "Unknown")))
    genre = html.escape(str(item.get("genre", "Unknown")))
    score = float(item.get("similarity", 0.0))
    return f"""
    <div style="
        background: linear-gradient(135deg, #1a1d2b 0%, #171a24 100%);
        border: 1px solid #2d3348;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: #f5f7fa;
    ">
      <div style="font-size: 18px; font-weight: 700; margin-bottom: 6px;">{title}</div>
      <div style="font-size: 14px; color: #d8dbe6; margin-bottom: 8px;">{desc}</div>
      <div style="font-size: 13px; color: #bfc7e6; margin-bottom: 8px;">Type: {industry} | Genre: {genre}</div>
      <div style="font-size: 13px; color: #89b4ff;">Similarity score: {score:.4f}</div>
    </div>
    """


def recommend_movies(industry: str, genre: str, show_all: bool) -> str:
    try:
        response = get_recommendations_by_preferences(
            top_k=5, industry_filter=industry, genre_filter=genre, show_all=show_all
        )
    except Exception as exc:
        return (
            "<div style='color:#ff8a8a;font-weight:600;'>"
            f"Request failed: {html.escape(str(exc))}"
            "</div>"
        )
    status = response.get("status")
    message = str(response.get("message", ""))
    results: List[Dict[str, object]] = response.get("results", [])

    if status != "success":
        return f"<div style='color:#ffd166;font-weight:600;'>{html.escape(message)}</div>"

    cards = "\n".join(_result_card(item) for item in results)
    return f"""
    <div style='color:#9be7a1;font-weight:600;margin-bottom:10px;'>{html.escape(message)}</div>
    {cards}
    """


CUSTOM_CSS = """
body, .gradio-container {
  background: #0f1117 !important;
  color: #f5f7fa !important;
}
#main-title {
  text-align: center;
  font-size: 2rem;
  font-weight: 800;
  margin-bottom: 8px;
}
"""


summary = get_dataset_summary()
industry_list = ", ".join(summary["industries"])
stats_text = f"Dataset: {summary['total_movies']} movies across {len(summary['industries'])} industries ({industry_list})."
status = check_endee_connection()
status_color = "#9be7a1" if status["status"] == "connected" else "#ffd166"

with gr.Blocks() as demo:
    gr.Markdown("<div id='main-title'>AI Movie Recommendation System</div>")
    gr.Markdown("Discover similar movies using AI semantic search with Endee vector database.")
    gr.Markdown(f"<div style='color:{status_color}; font-weight:600;'>{html.escape(status['message'])}</div>")
    gr.Markdown(f"<div style='color:#bfc7e6;'>{html.escape(stats_text)}</div>")

    with gr.Row():
        industry_choices = get_available_industries()
        genre_choices = get_available_genres()
        default_industry = industry_choices[0]
        default_genre = genre_choices[0]

        industry_input = gr.Dropdown(
            choices=industry_choices,
            value=default_industry,
            label="Movie Type / Industry",
            scale=2,
        )
        genre_input = gr.Dropdown(
            choices=genre_choices,
            value=default_genre,
            label="Genre / Section",
            scale=2,
        )
        show_all_input = gr.Checkbox(label="Show all recommendations", value=False, scale=1)
        run_btn = gr.Button("Get Recommendations", scale=1, variant="primary")

    output_html = gr.HTML(label="Recommendations")

    run_btn.click(
        fn=recommend_movies,
        inputs=[industry_input, genre_input, show_all_input],
        outputs=[output_html],
        show_progress="full",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
