"""Gradio frontend for AI movie recommendations (non-Streamlit option)."""

from __future__ import annotations

import html
from typing import List, Dict

import gradio as gr

from recommend import (
    get_all_movie_titles_by_industry,
    get_available_industries,
    get_recommendations,
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


def recommend_movies(movie_name: str, industry: str, show_all: bool) -> str:
    try:
        response = get_recommendations(
            movie_title=movie_name,
            top_k=5,
            industry_filter=industry,
            show_all=show_all,
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


def update_titles(industry: str):
    titles = get_all_movie_titles_by_industry(industry)
    if not titles:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=titles, value=titles[0])


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


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("<div id='main-title'>AI Movie Recommendation System</div>")
    gr.Markdown("Choose movie type (Hollywood, Bollywood, and more), then get top or all recommendations.")

    with gr.Row():
        industry_choices = get_available_industries()
        default_industry = industry_choices[0]
        initial_titles = get_all_movie_titles_by_industry(default_industry)

        industry_input = gr.Dropdown(
            choices=industry_choices,
            value=default_industry,
            label="Movie Type / Industry",
            scale=2,
        )
        movie_input = gr.Dropdown(
            choices=initial_titles,
            value=initial_titles[0] if initial_titles else None,
            label="Movie Name",
            allow_custom_value=True,
            scale=3,
        )
        show_all_input = gr.Checkbox(label="Show all recommendations", value=False, scale=1)
        run_btn = gr.Button("Get Recommendations", scale=1, variant="primary")

    output_html = gr.HTML(label="Recommendations")

    industry_input.change(fn=update_titles, inputs=[industry_input], outputs=[movie_input])

    run_btn.click(
        fn=recommend_movies,
        inputs=[movie_input, industry_input, show_all_input],
        outputs=[output_html],
        show_progress="full",
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
