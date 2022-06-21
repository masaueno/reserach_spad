import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def app():
    csv_input = st.file_uploader("spad_csv", type="csv")
    st.session_state["csv_input"] = csv_input
    do_denoise = st.checkbox("denoise")
    sampling_step = st.number_input("sampling step", min_value=1)
    showed_row = st.number_input("showd_ row", min_value=0, max_value=15)
    showed_column = st.number_input("showd_column", min_value=0, max_value=5)

    if st.session_state["csv_input"]:
        df_original = pd.read_csv(csv_input, skiprows=5)
        df_original = df_original.drop(columns="No.")

        df_sampled = df_original[::sampling_step]

        selected_receptor = df_sampled[f"R{showed_row}C{showed_column}"]
        if do_denoise:
            selected_receptor = selected_receptor.rolling(10).mean()

        fig = px.line(selected_receptor)
        st.plotly_chart(fig)
