import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import argrelmax, find_peaks


def app():
    """Sidebar options"""
    with st.sidebar:
        csv_input = st.file_uploader("spad_csv", type="csv")
        st.session_state["csv_input"] = csv_input
        sampleing_step = st.number_input("sampling step", min_value=1)

        should_select_receptor = st.checkbox("select_receptor")
        showed_row = -1
        showed_column = -1
        if should_select_receptor:
            showed_row = st.number_input("showed_ row", min_value=-1, max_value=15)
            showed_column = st.number_input("showed_column", min_value=-1, max_value=5)

        should_denoise = st.checkbox("denoise")
        if should_denoise:
            rolling = st.number_input("rolling", min_value=0, value=10)

        should_detect_peaks = st.checkbox("detect peak")
        if should_detect_peaks:
            prominence = st.number_input("prominence", min_value=0, value=200)

    """Figure View"""
    if st.session_state["csv_input"]:
        df_original = pd.read_csv(st.session_state["csv_input"], skiprows=5)
        df_original = df_original.drop(columns="No.")

        df_sampled = df_original[::sampleing_step]

        if should_select_receptor:
            if showed_row == -1 and showed_column == -1:
                pass
            elif showed_row == -1:
                df_sampled = df_sampled.filter(like=f"C{showed_column}", axis=1)
            elif showed_column == -1:
                df_sampled = df_sampled.filter(like=f"R{showed_row}C", axis=1)
            else:
                df_sampled = df_sampled[f"R{showed_row}C{showed_column}"]

        if should_denoise:
            df_sampled = df_sampled.rolling(rolling).median()  # type: ignore

        fig = px.line(df_sampled)

        if should_detect_peaks:
            array = df_sampled.to_numpy()
            if array.ndim == 1:
                array = array[:, np.newaxis]
            for receptor_id, receptor_array in enumerate(array.copy().T):
                peak_indices = find_peaks(receptor_array, prominence=prominence)[0]  # type: ignore
                fig.add_trace(
                    go.Scatter(
                        x=peak_indices,
                        y=array[peak_indices, receptor_id],
                        mode="markers",
                        marker=dict(size=8, color="red", symbol="cross"),
                        showlegend=False,
                    )
                )

        st.plotly_chart(fig)
