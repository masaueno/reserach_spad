import os
import pathlib
import sys

import pandas as pd
import plotly.graph_objects as go
from matplotlib import colors
from matplotlib import pyplot as plt

if len(sys.argv) < 3:
    print("more arg required")
    sys.exit()

input_file = sys.argv[1]

try:
    number1 = int(sys.argv[2])
except ValueError:
    print("arg2 must be int")
    sys.exit()

try:
    number2 = int(sys.argv[3])
except ValueError:
    print("arg3 must be int")
    sys.exit()

p_file = pathlib.Path(input_file)
if not os.path.isfile(input_file):
    print("file not found")
    sys.exit()

if number1 < 0 or number1 > 15:
    print("wrong arg2")
    sys.exit()

if number1 > number2:
    print("wrong args")
    sys.exit()

if number2 > 15:
    print("wrong arg3")
    sys.exit()

df = pd.read_csv(input_file, nrows=4)
delta = df.iat[3, 1]

df = pd.read_csv(input_file, skiprows=5)
fig = go.Figure()
xs = df["No."] * delta

y_shift = (number2 - number1 + 1) * 6

df_style = pd.read_csv(
    "style.csv", usecols=[0, 1], dtype={"background": object}, nrows=1
)

df_color = pd.read_csv(
    "style.csv", encoding="cp932", dtype=object, index_col=0, skiprows=2
)


for raw_number in range(number1, number2 + 1):
    for co_number in range(6):
        column_name = "R" + str(raw_number) + "C" + str(co_number)
        #        print(column_name)
        #            df[column_name] /= df.mean()[column_name]
        #            fig.add_trace(go.Scatter(x=xs, y=df[column_name]+y_shift, name=column_name))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=df[column_name] + y_shift,
                name=column_name,
                line=dict(
                    color=df_color.at["R" + str(raw_number), "C" + str(co_number)],
                    width=df_style.iat[0, 1],
                ),
            )
        )
        y_shift += 0
fig.update_layout(showlegend=True)
# fig.update_xaxes(rangeslider={"visible":True})
fig.update_layout(width=800, height=400 + 40 * (number2 - number1 + 1))
fig.update_layout(plot_bgcolor=df_style.iat[0, 0])
fig.show()
fig.write_html(p_file.stem + ".html")
