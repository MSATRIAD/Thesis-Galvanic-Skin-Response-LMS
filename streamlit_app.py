"""Entrypoint gabungan untuk visualisasi pipeline dan prediksi interaktif."""

import streamlit as st


st.set_page_config(
    page_title="Thesis Galvanic Skin Response LMS",
    page_icon="📊",
    layout="wide",
)

page = st.navigation(
    {
        "Penelitian": [
            st.Page(
                "app_streamlit/showcase.py",
                title="Proses Data sampai Evaluasi",
                icon=":material/analytics:",
                default=True,
            )
        ],
        "Penerapan": [
            st.Page(
                "app_streamlit/app.py",
                title="Prediksi CSV Baru",
                icon=":material/model_training:",
            )
        ],
    }
)
page.run()
