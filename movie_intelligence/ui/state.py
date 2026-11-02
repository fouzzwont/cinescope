"""Cached access to the dataset, model, recommender and TMDB client."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..dataset import load_movies
from ..modelling import ModelArtifact, ensure_model
from ..recommender import Recommender
from ..tmdb import TMDBClient


@st.cache_data(show_spinner=False)
def get_movies() -> pd.DataFrame:
    return load_movies()


@st.cache_resource(show_spinner="Indexing the catalogue ...")
def get_recommender() -> Recommender:
    return Recommender(get_movies())


@st.cache_resource(show_spinner="Training the revenue model ...")
def get_model() -> tuple[ModelArtifact, bool]:
    return ensure_model(get_movies())


@st.cache_resource(show_spinner=False)
def get_tmdb() -> TMDBClient:
    return TMDBClient(timeout=3.0)
