"""Mizan · ميزان: portfolio site and live demo.

Local:  streamlit run demo/streamlit_app.py
Cloud:  Streamlit Community Cloud, main file path = demo/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]   # `mizan` package (repo root) and `webui` (this folder)

import streamlit as st  # noqa: E402

from webui import components as ui  # noqa: E402

st.set_page_config(
    page_title="Mizan · Arabic banking intent classification",
    page_icon=str(HERE / "static" / "mizan-mark.svg"),
    layout="wide",
)
ui.inject_css()
st.logo(str(HERE / "static" / "mizan-logo.svg"), icon_image=str(HERE / "static" / "mizan-mark.svg"), size="large")

PAGES = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/demo.py", title="Live demo", url_path="demo"),
    st.Page("views/results.py", title="Results", url_path="results"),
    st.Page("views/how_it_works.py", title="How it works", url_path="how-it-works"),
    st.Page("views/case_study.py", title="Case study", url_path="case-study"),
]
st.navigation(PAGES, position="top").run()