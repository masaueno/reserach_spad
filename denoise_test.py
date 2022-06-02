import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df_old = pd.read_csv("data/spad_output/2205121705_flowCyto96(5).csv", skiprows=5)
df = df_old.drop(columns="No.")

step = 10
df_sampled = df[::step]

col_r7c2 = df_sampled["R7C2"]
col_r7c2_denoised = col_r7c2.rolling(10).mean()

fig = px.line(col_r7c2_denoised)
fig.show()
