import os
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colors
from matplotlib import pyplot as plt

# usage
# run SPAD96graph7.py inputfile  raw1 raw2 wind_size noise_supression_height peak_compression_flag total_laser
#

if len(sys.argv) < 8:
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

try:
    windsize = int(sys.argv[4])
except ValueError:
    print("arg4 must be int")
    sys.exit()

try:
    noisesup = float(sys.argv[5])
except ValueError:
    print("arg5 must be float")
    sys.exit()

try:
    compress = int(sys.argv[6])
except ValueError:
    print("arg6 must be 0 or 1")
    sys.exit()

try:
    num_laser = int(sys.argv[7])
except ValueError:
    print("arg7 must be 0 or 1 or 2")
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

if windsize < 1:
    print("wind size must be >0")
    sys.exit()

if noisesup < 0:
    print("noize supression height must be >0 or =0")
    sys.exit()

if compress > 1 or compress < 0:
    print("peak_compression_flag must be 0 or 1")
    sys.exit()

df = pd.read_csv(input_file, nrows=4)
delta = df.iat[3, 1]
total_laser = df.iat[2, 1]

if num_laser > total_laser - 1 or num_laser < 0:
    print("num_laser must be < #total_laser")
    sys.exit()

df_total = pd.read_csv(input_file, skiprows=5)

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

y_shift = (number2 - number1 + 1) * 6
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


for raw_number in reversed(range(number1, number2 + 1)):
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
                    color=df_color.at["R" + str(raw_number), "C" + str(co_number)],
                    width=df_style.iat[0, 1],
                ),
            )
        )
        y_shift += -1

    fig.update_layout(showlegend=True)
# fig.update_xaxes(rangeslider={"visible":True})
fig.update_layout(width=800, height=400 + 30 * (number2 - number1 + 1))
fig.update_layout(plot_bgcolor=df_style.iat[0, 0])
fig.update_layout(
    title={
        "text": "window size ="
        + str(windsize)
        + ",  noise supp height ="
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

fig.show()
fig.write_html(p_file.stem + ".html")
