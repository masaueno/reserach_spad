import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelmax, find_peaks
import matplotlib.pyplot as plt

df_old = pd.read_csv("data/spad_output/2205121705_flowCyto96(5).csv", skiprows=5)
df = df_old.drop(columns="No.")

step = 10
df_sampled = df[::step]

col_r7c2 = df_sampled["R7C2"]
col_r7c2_denoised = col_r7c2.rolling(10).mean()

array = col_r7c2_denoised.to_numpy()

# peak1 = find_peaks_cwt(array)
indices = find_peaks(array, prominence=200)[0]
print(indices)

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=array,
    mode='lines+markers',
    name='Original Plot'
))

fig.add_trace(go.Scatter(
    x=indices,
    y=[array[j] for j in indices],
    mode='markers',
    marker=dict(
        size=8,
        color='red',
        symbol='cross'
    ),
    name='Detected Peaks'
))

fig.show()