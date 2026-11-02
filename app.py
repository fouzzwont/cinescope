"""CineScope — movie intelligence dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from movie_intelligence.config import APP_ICON, APP_TITLE, MOVIES_PATH, tmdb_api_key
from movie_intelligence.ui import components, explore, overview, predict, recommend
from movie_intelligence.ui.state import get_movies

st.set_page_config(
    page_title=f"{APP_TITLE} · Movie intelligence",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
components.inject_css()

try:
    df = get_movies()
except FileNotFoundError:
    components.hero(
        "Setup required",
        "Dataset not found",
        "CineScope needs the cleaned TMDb catalogue before it can start.",
    )
    st.code("python scripts/build_dataset.py --download", language="bash")
    st.caption(f"Expected at: {MOVIES_PATH}")
    st.stop()

PAGES = {
    "Overview": overview.render,
    "Explore catalogue": explore.render,
    "Predict revenue": predict.render,
    "Recommendations": recommend.render,
}

with st.sidebar:
    st.markdown(
        """
        <div style="padding:.4rem 0 1rem 0">
          <div style="font-size:1.6rem;font-weight:800;letter-spacing:-.03em">
            <span style="color:#E50914">●</span> CineScope
          </div>
          <div style="color:#8B8F96;font-size:.78rem;margin-top:.1rem">
            Movie intelligence, not guesswork
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=list(PAGES),
        index=list(PAGES).index(st.query_params.get("page", ""))
        if st.query_params.get("page") in PAGES
        else 0,
        key="nav",
        label_visibility="collapsed",
    )
    if st.query_params.get("page") != page:
        st.query_params["page"] = page

    st.divider()
    st.caption("CATALOGUE")
    st.markdown(
        f"**{len(df):,}** films · **{df['release_year'].min():.0f}–{df['release_year'].max():.0f}**"
    )
    st.caption("DATA")
    st.markdown("TMDb 5000 (Kaggle)")
    st.caption("POSTERS")
    st.markdown("Live via TMDB API" if tmdb_api_key() else "Typographic fallback (no API key)")

    st.divider()
    with st.expander("About this project"):
        st.caption(
            "Built with Streamlit and scikit-learn. Revenue predictions are "
            "statistical estimates from historical metadata, not financial advice. "
            "See the README for data provenance and modelling notes."
        )

PAGES[page](df)
