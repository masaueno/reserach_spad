import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("data/spad_output/2205121705_flowCyto96(5).csv", skiprows=5)
df_new = df.drop(columns="No.")

step = 10
df_selected = df_new[::step]


fig = px.line(df_new[25000:30000])
fig.show()
