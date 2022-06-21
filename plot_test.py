import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("data/spad_output/2206021817_flowCyto96.csv", skiprows=5)
df_new = df.drop(columns="No.")

step = 3
df_red = df_new[0::step]
df_blue = df_new[1::step]
df_green = df_new[2::step]

fig = px.line(df_red)
fig.show()

fig = px.line(df_blue)
fig.show()

fig = px.line(df_green)
fig.show()
