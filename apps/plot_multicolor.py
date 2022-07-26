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
    # run SPAD96graph7.py inputfile  raw1 raw2 wind_size noise_supression_height peak_compression_flag

    """Sidebar options"""
    with st.sidebar:
        st.session_state["csv_input"] = st.file_uploader("spad_csv", type="csv")

        raw1 = st.number_input("raw1", min_value=0, max_value=15, value=0)
        raw2 = st.number_input("raw2", min_value=0, max_value=15, value=15)
        windsize = st.number_input("wind_size", value=30)
        noisesup = st.number_input("noisesup", value=2.5)
        compress = st.number_input("compress", value=1)
        num_laser = st.number_input("num_laser", value=1)

        should_detect_peaks = st.checkbox("detect peak")
        if should_detect_peaks:
            prominence = st.number_input("prominence", min_value=0, value=100)

    """Figure View"""
    if st.session_state["csv_input"]:
        df_total = pd.read_csv(st.session_state["csv_input"], skiprows=5)
        delta = df.iat[3, 1]
        total_laser = df.iat[2, 1]

        if total_laser == 1:
            df = df_total
        elif total_laser == 2:
            if num_laser == 0:
                df = df_total[0::2]
            elif num_laser == 1:
                df = df_total[1::2]
        elif total_laser == 3:
            if num_laser == 0:
                df = df_total[0::3]
            elif num_laser == 1:
                df = df_total[1::3]
            elif num_laser == 2:
                df = df_total[2::3]

        fig = go.Figure()

        xs = df["No."].iloc[windsize // 2 :] * delta

        y_shift = (raw2 - raw1 + 1) * 6
        df_style = pd.read_csv(
            "style.csv", usecols=[0, 1], dtype={"background": object}, nrows=1
        )
        df_color = pd.read_csv(
            "style.csv", encoding="cp932", dtype=object, index_col=0, skiprows=2
        )

        def comp(x):
            if x > 1:
                return 4 * np.log10(x) + 1
            else:
                return x

        for raw_number in reversed(range(raw1, raw2 + 1)):
            for co_number in range(6):
                column_name = "R" + str(raw_number) + "C" + str(co_number)

                # moving average
                yc = df[column_name].rolling(windsize).sum()
                data = np.array(yc.iloc[windsize:])

                hist, bin_edges = np.histogram(data, bins="auto")
                hist_df = pd.DataFrame(columns=["start", "end", "count"])
                for idx, val in enumerate(hist):
                    start = round(bin_edges[idx], 2)
                    end = round(bin_edges[idx + 1], 2)
                    hist_df.loc[idx] = [start, end, val]

                imax = hist_df["count"].idxmax()
                ave = (hist_df.iloc[imax, 0] + hist_df.iloc[imax, 1]) / 2
                sigma = ave**0.5

                # noise supression
                yc = yc.where(
                    (ave - noisesup * sigma > yc) | (ave + noisesup * sigma < yc), ave
                )
                yc = yc.where(yc < ave + noisesup * sigma, yc - noisesup * sigma)
                yc = yc.where(yc > ave - noisesup * sigma, yc + noisesup * sigma)
                yc -= ave
                yc /= 6 * sigma
                if compress == 1:
                    yc = yc.map(comp)
                #            yc /= yc.mean()
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=yc.iloc[windsize:] + y_shift,
                        name=column_name,
                        line=dict(
                            color=df_color.at[
                                "R" + str(raw_number), "C" + str(co_number)
                            ],
                            width=df_style.iat[0, 1],
                        ),
                    )
                )

                # Detect and plot peaks
                if should_detect_peaks:
                    array = (yc.iloc[windsize:] + y_shift).to_numpy()
                    peak_indices = find_peaks(array * 100, prominence=prominence)[0]  # type: ignore
                    fig.add_trace(
                        go.Scatter(
                            x=xs[peak_indices],
                            y=array[peak_indices],
                            mode="markers",
                            marker=dict(size=8, color="red", symbol="cross"),
                            showlegend=False,
                        )
                    )

                y_shift += -1

            fig.update_layout(showlegend=True)
        # fig.update_xaxes(rangeslider={"visible":True})
        fig.update_layout(width=800, height=400 + 30 * (raw2 - raw1 + 1))
        fig.update_layout(plot_bgcolor=df_style.iat[0, 0])
        fig.update_layout(
            title={
                "text": "window size ="
                + str(windsize)
                + ",  noise suppression height ="
                + str(noisesup)
                + ", peak compression ="
                + str(compress)
                + ", laser num ="
                + str(num_laser),
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )

        st.plotly_chart(fig)
