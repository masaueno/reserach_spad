# ----------------------------------------
# ----------------------------------------
# ----------------------------------------
#
#  This program analyses the output of 96-SPAD array ouput
#  	ver. 1.0
#  	released: 21.09.25
#
# 	Lab Arco Ltd.
#
#
#  usage
#  run cell_extraction.py "inputfile"  wind_size
#  with style.csv in the same directory
# ----------------------------------------
# ----------------------------------------
# ----------------------------------------


import math
import os
import pathlib
import sys
import winsound

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colors
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy import signal

if len(sys.argv) < 3:
    print("more arg required")
    sys.exit()

input_file = sys.argv[1]

try:
    windsize = int(sys.argv[2])
except ValueError:
    print("arg2 must be int")
    sys.exit()

if not os.path.isfile(input_file):
    print("file not found")
    sys.exit()

if windsize < 1:
    print("wind size must be >0")
    sys.exit()

number1 = 0
number2 = 15
compress = 1
noisesup = 6

p_file = pathlib.Path(input_file)
df = pd.read_csv(input_file, nrows=4)
df = pd.read_csv(input_file, skiprows=5)
xs = df["No."]


def comp(x):
    if x > 1:
        return 4 * np.log10(x) + 1
    else:
        return max(0, x)


df_style = pd.read_csv(
    "style.csv", usecols=[0, 1], dtype={"background": object}, nrows=1
)
df_color = pd.read_csv(
    "style.csv", encoding="cp932", dtype=object, index_col=0, skiprows=2
)

# ----------------------------------------
# ----------------------------------------  signal conditioning
# ----------------------------------------
def sig_con():
    print("signal conditioning process start")
    for row_number in reversed(range(number1, number2 + 1)):
        for co_number in range(6):
            column_name = "R" + str(row_number) + "C" + str(co_number)

            # moving average
            yc = df[column_name].rolling(windsize, min_periods=1).mean()
            data = np.array(yc)

            hist, bin_edges = np.histogram(data, bins="auto")
            hist_df = pd.DataFrame(columns=["start", "end", "count"])
            for idx, val in enumerate(hist):
                start = round(bin_edges[idx], 2)
                end = round(bin_edges[idx + 1], 2)
                hist_df.loc[idx] = [start, end, val]

            imax = hist_df["count"].idxmax()
            ave = (hist_df.iloc[imax, 0] + hist_df.iloc[imax, 1]) / 2
            sigma = ave**0.5 / windsize**0.5

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
            else:
                yc = yc.clip(0)

            df[column_name] = yc
    # ----------------------------------------
    print("signal conditioning process end")
    return


# ----------------------------------------
# ----------------------------------------

# ----------------------------------------
# ----------------------------------------  drawing
# ----------------------------------------
def draw_graph(type):
    fig = go.Figure()
    y_shift = (number2 - number1 + 1) * 6
    x_shift = 0
    for row_number in reversed(range(number1, number2 + 1)):
        for co_number in range(6):
            column_name = "R" + str(row_number) + "C" + str(co_number)
            id = (15 - row_number) * 6 + co_number

            if type == 0:
                fig.add_trace(
                    go.Scatter(
                        x=xs - x_shift,
                        y=df[column_name] + y_shift,
                        name=column_name,
                        line=dict(
                            color=df_color.at[
                                "R" + str(row_number), "C" + str(co_number)
                            ],
                            width=df_style.iat[0, 1],
                        ),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=xs - x_shift,
                        y=df2[column_name] + y_shift,
                        name=column_name,
                        line=dict(
                            color=df_color.at[
                                "R" + str(row_number), "C" + str(co_number)
                            ],
                            width=df_style.iat[0, 1],
                        ),
                    )
                )
            y_shift += -4

            fig.update_layout(showlegend=True)
    # fig.update_xaxes(rangeslider={"visible":True})
    fig.update_layout(width=600, height=300 + 20 * (number2 - number1 + 1))
    fig.update_layout(plot_bgcolor=df_style.iat[0, 0])
    fig.update_layout(
        title={
            "text": "window size ="
            + str(windsize)
            + ",  noise suppression height ="
            + str(noisesup)
            + ", peak compression ="
            + str(compress),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )

    fig.show()


# ----------------------------------------

# ---------------------------------------------------
# ---------------------------------------------------
#  draw side by side
# ---------------------------------------------------
# ---------------------------------------------------


def draw_graph2():
    #    fig = go.Figure()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("original", "cell deleted"),
        row_heights=[2],
        column_widths=[0.5, 0.5],
        shared_xaxes=True,
        shared_yaxes=True,
    )
    y_shift = 12
    df_style = pd.read_csv(
        "style.csv", usecols=[0, 1], dtype={"background": object}, nrows=1
    )
    df_color = pd.read_csv(
        "style.csv", encoding="cp932", dtype=object, index_col=0, skiprows=2
    )

    for row_number in reversed(range(16)):
        for co_number in range(6):
            column_name = "R" + str(row_number) + "C" + str(co_number)
            yc = df[column_name]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=yc + y_shift,
                    name=column_name,
                    line=dict(
                        color=df_color.at["R" + str(row_number), "C" + str(co_number)],
                        width=df_style.iat[0, 1],
                    ),
                ),
                row=1,
                col=1,
            )
            yc = df2[column_name]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=yc + y_shift,
                    name=column_name,
                    line=dict(
                        color=df_color.at["R" + str(row_number), "C" + str(co_number)],
                        width=df_style.iat[0, 1],
                    ),
                ),
                row=1,
                col=2,
            )
            y_shift += -8
        fig.update_layout(showlegend=True)

    fig.update_layout(width=800, height=600)
    fig.update_layout(
        plot_bgcolor=df_style.iat[0, 0],
        showlegend=False,
    )

    fig.show()
    #    fig.write_html(p_file.stem+".html")
    return


# ---------------------------------------------------
# ---------------------------------------------------


# calculate representative shift
# ----------------------------------------
# ----------------------------------------
def rep_corr():
    global rep_shift
    corr_list = []
    for x_shift in range(5, 300, 5):
        corr = 0
        for row_number in range(0, 16):
            for co_number in range(5):
                column_ref = "R" + str(row_number) + "C" + str(co_number)
                column_low = "R" + str(row_number) + "C" + str(co_number + 1)
                for x in range(2048 - x_shift):
                    corr += df2[column_ref][x] * df2[column_low][x + x_shift]
        corr_list.append(corr)
    rep_shift = np.argmax(corr_list) * 5 + 5
    print("rep_shift", rep_shift)

    return


# ----------------------------------------


# correlation -> shift
# Two rows prosessing
# ----------------------------------------  correlation
# ----------------------------------------
def corr_shift(upper_row):
    global corshift
    corr_list = []
    min_shift = int(rep_shift * 0.7)
    max_shift = int(rep_shift * 3.0)
    for x_shift in range(min_shift, max_shift, 10):
        corr = 0
        for row_number in range(upper_row, upper_row + 1):
            for co_number in range(5):
                column_ref = "R" + str(row_number) + "C" + str(co_number)
                column_low = "R" + str(row_number) + "C" + str(co_number + 1)
                for x in range(2048 - x_shift):
                    corr += df2[column_ref][x] * df2[column_low][x + x_shift]
        #    print(x_shift, corr)
        corr_list.append(corr)
    shift_c = np.argmax(corr_list) * 10 + min_shift
    print("shift_c", shift_c)

    corr_list = []
    for x_shift in range(max(1, shift_c - 10), shift_c + 10):
        corr = 0
        for row_number in range(upper_row - 1, upper_row + 1):
            for co_number in range(5):
                column_ref = "R" + str(row_number) + "C" + str(co_number)
                column_low = "R" + str(row_number) + "C" + str(co_number + 1)
                for x in range(2048 - x_shift):
                    corr += df2[column_ref][x] * df2[column_low][x + x_shift]
        #    print(x_shift, corr)
        corr_list.append(corr)
    shift_f = np.argmax(corr_list)
    corshift = max(1, shift_c - 10) + shift_f
    print("correlation max shift", corshift)
    return


# ----------------------------------------


# peak_row
# peak spread
# create peak_row(spad idex)
# prominent を調整すること
# ----------------------------------------  peak search
def create_peak_row(upper_row):
    global peak_row
    for row_d in range(2):
        for co_number in range(6):
            row_number = upper_row - row_d
            column_name = "R" + str(row_number) + "C" + str(co_number)
            yy = df2[column_name].values
            maxid = signal.find_peaks(
                yy, distance=1 + int(rep_shift / 4), prominence=0.8
            )
            #                print("co_number", co_number)
            #                print(maxid)
            peakrow = []

            for index, peak in enumerate(maxid[0]):
                #                    print ("peak", peak)
                #                    print ("index", index)
                #                    print (maxid[1]['left_bases'][index])

                if index > 0:
                    pre_right = maxid[1]["right_bases"][index - 1]
                else:
                    pre_right = 0
                left = max(pre_right, maxid[1]["left_bases"][index])

                if pre_right < maxid[1]["left_bases"][index]:
                    left = maxid[1]["left_bases"][index]
                elif pre_right > maxid[0][index]:
                    left = maxid[1]["left_bases"][index]
                else:
                    left = pre_right

                if index < len(maxid[0]) - 1:
                    post_left = maxid[1]["left_bases"][index + 1]
                else:
                    post_left = 2047

                if post_left > maxid[1]["right_bases"][index]:
                    right = maxid[1]["right_bases"][index]
                elif post_left < maxid[0][index]:
                    right = maxid[1]["right_bases"][index]
                else:
                    right = post_left

                peak_st = [left, peak, right]
                #                    print(peak_st)
                peakrow.append(peak_st)
            peak_row.append(peakrow)
    return


# ----------------------------------------


import copy
# cell_row
# create cell row
# ----------------------------------------
import re


def create_cell_row(upper_row):
    global cell_row
    global peak_row
    # ----------------------------------------  start peak spread,
    for row_number in reversed(range(upper_row - 1, upper_row + 1)):
        cellrow = []
        for co_number in range(6):
            column_ref = "R" + str(row_number) + "C" + str(co_number)
            id = (upper_row - row_number) * 6 + co_number
            while len(peak_row[id]) > 0:
                peaks = []
                peak = peak_row[id][0]
                #                    print("id, peak", id, peak)
                peak_row[id].remove(peak)
                peak_st = [column_ref]
                peak_st.extend(peak)
                peaks.append(peak_st)
                # ----------------------------------------  peak grouping
                pre_peakidx = peak[1]
                for column_next in range(1, 6 - co_number):
                    column_ref2 = (
                        "R" + str(row_number) + "C" + str(co_number + column_next)
                    )
                    num = id + column_next
                    if len(peak_row[num]) < 1:
                        break
                    sq = []
                    for peak in peak_row[num]:
                        sq.extend([peak[1]])

                    idx = np.abs(np.asarray(sq) - (pre_peakidx + corshift)).argmin()
                    dis = np.abs(np.asarray(sq) - (pre_peakidx + corshift)).min()
                    cellwidth = peak_row[num][idx][2] - peak_row[num][idx][0]
                    if dis < 1 + corshift * 0.15 + cellwidth * 0.1:
                        peak = peak_row[num][idx]
                        peakidx = peak_row[num][idx][1]
                        peak_row[num].remove(peak)
                        pre_peakidx = peakidx
                        peak_st = [column_ref2]
                        peak_st.extend(peak)
                        #                            print("peak_st", peak_st)
                        peaks.append(peak_st)
                    else:
                        break
                left_min = 2047
                right_max = 0
                co_first = 5
                co_last = 0
                for peak_st in peaks:
                    result = re.findall(r"\d+", peak_st[0])
                    column_name = peak_st[0]
                    left = peak_st[1]
                    right = peak_st[3]
                    row_number2 = int(result[0])
                    co_number2 = int(result[1])
                    left_min = min(left_min, left)
                    right_max = max(right_max, right)
                    co_first = min(co_first, co_number2)
                    co_last = max(co_last, co_number2)
                left_most = left_min - corshift * (co_first)
                right_most = right_max + corshift * (5 - co_last)
                cell_st = [peaks, [left_most, right_most], rep_shift / corshift]
                cellrow.append(cell_st)
        cell_row.append(cellrow)
    return


# ----------------------------------------
# ---------------------------------------------------


# ---------------------------------------------------
# verify cell_row and create cell_row_new
# ---------------------------------------------------
def create_cell_row_new(upper_row):
    global cell_row_new
    # ---------------------------------------------------
    # function cell append
    # ---------------------------------------------------
    def addcell(peaks):
        left_min = 2048
        right_max = 0
        co_first = 5
        co_last = 0
        for peak in peaks:
            result = re.findall(r"\d+", peak[0])
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number2 = int(result[0])
            co_number2 = int(result[1])

            left_min = min(left_min, left)
            right_max = max(right_max, right)
            co_first = min(co_first, co_number2)
            co_last = max(co_last, co_number2)
        left_most = left_min - corshift * (co_first)
        right_most = right_max + corshift * (5 - co_last)
        cell_st = [peaks, [left_most, right_most], rep_shift / corshift]
        cellrow_new.append(cell_st)
        return cellrow_new

    # ---------------------------------------------------
    global cell_row_new

    for cellrow in cell_row:
        #        print("cellrow start")
        cellrow_new = []
        for cell in cellrow:
            cell_new = []
            #            print(cell)
            top = 0
            slope = 1
            peaks = []
            for peak in cell[0]:
                if (slope > 0) & (df2[peak[0]][peak[2]] >= top):
                    peaks.append(peak)
                    top = df2[peak[0]][peak[2]]
                    slope = 1
                elif df2[peak[0]][peak[2]] < top:
                    peaks.append(peak)
                    top = df2[peak[0]][peak[2]]
                    slope = -1
                elif df2[peak[0]][peak[2]] >= top:
                    print("divide")
                    cellrow_new = addcell(peaks)
                    peaks = []
                    peaks.append(peak)
                    top = df2[peak[0]][peak[2]]
                    slope = 1
            cellrow_new = addcell(peaks)
        cell_row_new.append(cellrow_new)

    #    print([len(row) for row in cell_row])
    #    print([len(row) for row in cell_row_new])
    return


# ---------------------------------------------------
# ---------------------------------------------------

# ---------------------------------------------------
# ---------------------------------------------------
# draw cell_row
# ---------------------------------------------------
# ---------------------------------------------------
def draw_cell(cell_list):
    for cellrow in cell_list:
        for cell in cellrow:
            #            print(cell)
            plt.figure(figsize=(3, 2))
            for peak in cell[0]:
                result = re.findall(r"\d+", peak[0])
                column_name = peak[0]

                row_number = int(result[0])
                co_number = int(result[1])
                #            id = row_number*6 - co_number
                id = row_number * 6
                y_shift = id
                left = cell[1][0]
                right = cell[1][1]
                plt.plot(
                    xs.iloc[left : right + 1],
                    df2[column_name].iloc[left : right + 1] + y_shift,
                    color="blue",
                )

                left = peak[1]
                right = peak[3]

                plt.plot(
                    xs.iloc[left : right + 1],
                    df2[column_name].iloc[left : right + 1] + y_shift,
                    color="red",
                )
            #        plt.xlim(cell[1])
            plt.ylim(bottom=y_shift)
            plt.show()
    return


# ---------------------------------------------------
# ---------------------------------------------------

# ---------------------------------------------------
# ---------------------------------------------------
# pair finding and create list pairs
# ---------------------------------------------------
def create_pairs():
    global pairs
    global cell_row_new
    pairs = []
    for cell1 in cell_row_new[0]:
        for peak1 in cell1[0]:
            result = re.findall(r"\d+", peak1[0])
            co_number = int(result[1])
            if co_number == 5:
                for cell2 in cell_row_new[1]:
                    for peak2 in cell2[0]:
                        result = re.findall(r"\d+", peak2[0])
                        co_number = int(result[1])
                        if co_number == 0:
                            if (
                                min(cell1[1][1], cell2[1][1])
                                - max(cell1[1][0], cell2[1][0])
                            ) > corshift * 5:
                                pairs.append([cell1, cell2])
                                cell_row_new[0].remove(cell1)
                                cell_row_new[1].remove(cell2)
    return


# ---------------------------------------------------
# ---------------------------------------------------

# pair verification and correction
# ---------------------------------------------------
# ---------------------------------------------------
def verify_pairs():
    global pairs
    global cell_row_new
    i = 0
    for pair in pairs:
        i += 1
        #        print(i)
        #        print(pair[0])
        #        print(pair[1])
        upeaks = []
        for peak in pair[0][0]:
            #        print(peak[0])
            result = re.findall(r"\d+", peak[0])
            #        print(result)
            column_name = peak[0]
            uedge = df2[peak[0]].iloc[peak[2]]
            upeaks.append(uedge)
            co_number = int(result[1])

        if len(upeaks) > 0:
            #        print(upeaks)
            umaxid = co_number - len(upeaks) + upeaks.index(max(upeaks)) + 1
            umax = max(upeaks)
        else:
            umaxid = 5
        #    print("umaxid", umaxid)

        lpeaks = []
        for peak in pair[1][0]:
            #        print(peak[0])
            result = re.findall(r"\d+", peak[0])
            #        print(result)
            column_name = peak[0]
            lpeaks.append(df2[peak[0]].iloc[peak[2]])
            co_number = int(result[1])
            if co_number == 0:
                ledge = df2[peak[0]].iloc[peak[2]]

        if len(lpeaks) > 0:
            #        print(lpeaks)
            lmaxid = co_number - len(lpeaks) + lpeaks.index(max(lpeaks)) + 1
            lmax = max(lpeaks)
        else:
            lmaxid = 0
        #    print("lmaxid", lmaxid)

        if (umaxid < 5) & (lmaxid > 0):
            print("irregular pair removed")
            pairs.remove(pair)
            cell_row_new[0].append(pair[0])
            cell_row_new[1].append(pair[1])
        elif (umaxid < 5) & (uedge * 1.1 < ledge):
            print("irregular pair removed")
            pairs.remove(pair)
            cell_row_new[0].append(pair[0])
            cell_row_new[1].append(pair[1])

    return


# ---------------------------------------------------
# ---------------------------------------------------

# ---------------------------------------------------
# ---------------------------------------------------
# pair drawing
def draw_pairs():
    numpair = len(pairs)
    fig, ax = plt.subplots(
        nrows=math.ceil(numpair / 4), ncols=4, figsize=(20, 5 * math.ceil(numpair / 4))
    )
    ax = ax.flatten()

    for pair, i in enumerate(pairs):
        plt.figure(figsize=(2, 1.5))
        print(i)
        #        print(pair[0])
        #        print(pair[1])

        for peak in pair[0][0]:
            #        print(peak[0])
            result = re.findall(r"\d+", peak[0])
            #        print(result)
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = row_number * 6 - co_number
            #        y_shift = id
            y_shift = 0
            ax[i].plot(
                xs.iloc[left - 1 : right + 2],
                df2[column_name].iloc[left - 1 : right + 2] + y_shift,
                color="blue",
            )

        for peak in pair[1][0]:
            #        print(peak[0])
            result = re.findall(r"\d+", peak[0])
            #        print(result)
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = row_number * 6 - co_number
            #        y_shift = id
            y_shift = 0
            ax[i].plot(
                xs.iloc[left - 1 : right + 2],
                df2[column_name].iloc[left - 1 : right + 2] + y_shift,
                color="red",
            )

        plt.show()
    return


# ---------------------------------------------------
# ---------------------------------------------------

# ---------------------------------------------------
# ---------------------------------------------------
# cell confirmation and deletion from df2
# create row_cells, row_pairs
# ---------------------------------------------------
# ---------------------------------------------------
def cell_confirm():
    global df2
    global confirmed_cells, confirmed_pairs
    global cell_row_new

    found_new = 0
    #    for pair in pairs:
    for pair in pairs[:]:
        for peak in pair[0][0]:
            column_name = peak[0]
            result = re.findall(r"\d+", peak[0])
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = row_number * 6 + co_number + 1
            #            df2[column_name].iloc[left:right+1] = 0
            df2.iloc[left : right + 1, id] = 0
        for peak in pair[1][0]:
            column_name = peak[0]
            result = re.findall(r"\d+", peak[0])
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = row_number * 6 + co_number + 1
            #            df2[column_name].iloc[left:right+1] = 0
            df2.iloc[left : right + 1, id] = 0
        found_new += 1
        confirmed_pairs.append(pair)
        #        print("confirmed pair shift", pair[0][2], pair[1][2])
        #        print("pair", pair[0],pair[1])
        pairs.remove(pair)

    #    import pdb; pdb.set_trace()
    #    for cell in cell_row_new[0]:
    for cell in cell_row_new[0][:]:
        if len(cell[0]) > 1:
            for peak in cell[0]:
                column_name = peak[0]
                result = re.findall(r"\d+", peak[0])
                left = peak[1]
                right = peak[3]
                row_number = int(result[0])
                co_number = int(result[1])
                id = row_number * 6 + co_number + 1
                df2.iloc[left : right + 1, id] = 0
            found_new += 1
            confirmed_cells.append(cell)
            cell_row_new[0].remove(cell)

    #    for cell in cell_row_new[1]:
    for cell in cell_row_new[1][:]:
        #        import pdb; pdb.set_trace()
        if len(cell[0]) > 1:
            for peak in reversed(cell[0]):
                column_name = peak[0]
                result = re.findall(r"\d+", peak[0])
                left = peak[1]
                right = peak[3]
                row_number = int(result[0])
                co_number = int(result[1])
                id = row_number * 6 + co_number + 1
                df2.iloc[left : right + 1, id] = 0
            found_new += 1
            confirmed_cells.append(cell)
            cell_row_new[1].remove(cell)

    return found_new


# ---------------------------------------------------
# ---------------------------------------------------

# ---------------------------------------------------
# ---------------------------------------------------
# df2 drawing
# ---------------------------------------------------
# ---------------------------------------------------
from plotly.subplots import make_subplots


def draw_df2(upper_row):
    #    fig = go.Figure()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("original", "cell deleted"),
        row_heights=[0.4],
        column_widths=[0.5, 0.5],
        shared_xaxes=True,
        shared_yaxes=True,
    )
    y_shift = 12
    df_style = pd.read_csv(
        "style.csv", usecols=[0, 1], dtype={"background": object}, nrows=1
    )
    df_color = pd.read_csv(
        "style.csv", encoding="cp932", dtype=object, index_col=0, skiprows=2
    )

    for row_number in reversed(range(upper_row - 1, upper_row + 1)):
        for co_number in range(6):
            column_name = "R" + str(row_number) + "C" + str(co_number)
            yc = df[column_name]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=yc + y_shift,
                    line=dict(
                        color=df_color.at["R" + str(row_number), "C" + str(co_number)],
                        width=df_style.iat[0, 1],
                    ),
                ),
                row=1,
                col=1,
            )
            yc = df2[column_name]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=yc + y_shift,
                    line=dict(
                        color=df_color.at["R" + str(row_number), "C" + str(co_number)],
                        width=df_style.iat[0, 1],
                    ),
                ),
                row=1,
                col=2,
            )
            y_shift += -3
        fig.update_layout(showlegend=True)

    fig.update_layout(width=600, height=300)
    fig.update_layout(
        plot_bgcolor=df_style.iat[0, 0],
        showlegend=False,
    )

    fig.show()
    fig.write_html(p_file.stem + ".html")
    return


# ---------------------------------------------------
# ---------------------------------------------------

# ---------------------------------------------------
# ---------------------------------------------------
# df drawing
# ---------------------------------------------------
# ---------------------------------------------------
def draw_df(upper_row):
    fig = go.Figure()

    y_shift = 12
    df_style = pd.read_csv(
        "style.csv", usecols=[0, 1], dtype={"background": object}, nrows=1
    )
    df_color = pd.read_csv(
        "style.csv", encoding="cp932", dtype=object, index_col=0, skiprows=2
    )

    for row_number in reversed(range(upper_row - 1, upper_row + 1)):
        for co_number in range(6):
            column_name = "R" + str(row_number) + "C" + str(co_number)
            yc = df[column_name]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=yc + y_shift,
                    name=column_name,
                    line=dict(
                        color=df_color.at["R" + str(row_number), "C" + str(co_number)],
                        width=df_style.iat[0, 1],
                    ),
                )
            )
            y_shift += -10
        fig.update_layout(showlegend=True)

    fig.update_layout(width=500, height=400)
    fig.update_layout(plot_bgcolor=df_style.iat[0, 0])

    fig.show()
    return


# ---------------------------------------------------
# ---------------------------------------------------
# remaining check
# ---------------------------------------------------
# ---------------------------------------------------
def check_remaining():
    relist = []
    if len(cell_row_new[0]) > 1:
        for cell in cell_row_new[0]:
            for peak in cell[0]:
                result = re.findall(r"\d+", peak[0])
                co_number = int(result[1])
                relist.append(co_number)
        relist.sort()  # assending oredered
        relist_r = np.roll(relist, 1)
        #        import pdb; pdb.set_trace()
        print("relist", relist)
        if 1 in (relist - relist_r):
            remaining = 1
        else:
            remaining = 0
    else:
        remaining = 0
    return remaining


# ---------------------------------------------------
# ---------------------------------------------------


def draw_confirmed_pairs():
    numpair = len(confirmed_pairs)
    print("numpair", numpair)
    fig, ax = plt.subplots(
        nrows=math.ceil(numpair / 4), ncols=4, figsize=(10, 2 * math.ceil(numpair / 4))
    )
    ax = ax.flatten()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for i, pair in enumerate(confirmed_pairs):
        delta = pair[0][1][1] - pair[1][1][0]
        sum_sig = 0
        for peak in pair[0][0]:
            #        print(peak[0])
            result = re.findall(r"\d+", peak[0])
            #        print(result)
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = row_number * 6 - co_number
            y_shift = id
            sum_sig += right - left
            ax[i].plot(
                xs.iloc[left - 1 : right + 2],
                df[column_name].iloc[left - 1 : right + 2] + y_shift,
                color="blue",
            )

        for peak in pair[1][0]:
            #        print(peak[0])
            result = re.findall(r"\d+", peak[0])
            #        print(result)
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = row_number * 6 - co_number
            y_shift = id
            sum_sig += right - left
            ax[i].plot(
                xs.iloc[left - 1 : right + 2],
                df[column_name].iloc[left - 1 : right + 2] + y_shift,
                color="red",
            )

        sum_sig = int(((sum_sig / delta) ** 0.5) * 20)
        ax[i].set_title("dia:" + str(sum_sig), x=0.2)
    #        print("sum_sig", sum_sig)

    #        print("sum_sig/delta", sum_sig/delta)
    #        print("delta", delta)
    #        print ("sum_sig", sum_sig)

    plt.show()
    return


# ---------------------------------------------------
# ---------------------------------------------------

# ---------------------------------------------------
# ---------------------------------------------------
# draw_confirmed_cells
# ---------------------------------------------------
# ---------------------------------------------------
def draw_confirmed_cells():
    numcell = len(confirmed_cells)
    numpic = math.ceil(numcell / 20)
    print("numcells", numcell)
    #    print("numpic", numpic)
    for j in range(numpic):
        fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(10, 10))
        ax = ax.flatten()
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        i = 0
        for n, cell in enumerate(confirmed_cells):
            if (j * 20 <= n) & (n <= (j + 1) * 20 - 1):
                sum_sig = 0
                delta = cell[1][1] - cell[1][0]
                for peak in cell[0]:
                    result = re.findall(r"\d+", peak[0])
                    column_name = peak[0]
                    #            print(peak[0])
                    row_number = int(result[0])
                    co_number = int(result[1])
                    id = row_number * 6 - co_number
                    #            id = row_number*6
                    y_shift = id
                    #            plt.plot(xs.iloc[left:right+1], df[column_name].iloc[left:right+1]+y_shift,
                    #                     color = "blue" )

                    left = peak[1]
                    right = peak[3]

                    ax[i].plot(
                        xs.iloc[left : right + 1],
                        df[column_name].iloc[left : right + 1] + y_shift,
                        color="red",
                    )
                    sum_sig += right - left
                sum_sig = int(((sum_sig / delta) ** 0.5) * 20)
                ax[i].set_title("dia:" + str(sum_sig), x=0.2)
                ax[i].set_xlim(cell[1])
                ax[i].set_ylim(bottom=y_shift)
                i += 1

        plt.show()

    return


# ---------------------------------------------------
# ---------------------------------------------------


def final_draw():

    import re

    fig = go.Figure()
    cmap = plt.get_cmap("tab20")
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ----------------------------------------  group range estimate

    cell_num = 0
    for cell in confirmed_cells:
        for peak in cell[0]:
            result = re.findall(r"\d+", peak[0])
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = (15 - row_number) * 6 + co_number
            y_shift = 200 - id * 5
            xlist = xs.iloc[left : right + 1]
            ylist = df[column_name].iloc[left : right + 1] + y_shift
            fig.add_trace(
                go.Scatter(
                    x=xlist,
                    y=ylist,
                    name="cell" + str(cell_num),
                    mode="lines",
                    line=dict(color=cycle[int(cell_num % 10)]),
                )
            )
            fig.update_layout(showlegend=False)
        cell_num += 1

    for pair in confirmed_pairs:
        for peak in pair[0][0]:
            result = re.findall(r"\d+", peak[0])
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = (15 - row_number) * 6 + co_number
            y_shift = 200 - id * 5
            xlist = xs.iloc[left : right + 1]
            ylist = df[column_name].iloc[left : right + 1] + y_shift
            fig.add_trace(
                go.Scatter(
                    x=xlist,
                    y=ylist,
                    name="cell" + str(cell_num),
                    mode="lines",
                    line=dict(color=cycle[int(cell_num % 10)]),
                )
            )
            fig.update_layout(showlegend=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

        for peak in pair[1][0]:
            result = re.findall(r"\d+", peak[0])
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = (15 - row_number) * 6 + co_number
            y_shift = 200 - id * 5
            xlist = xs.iloc[left : right + 1]
            ylist = df[column_name].iloc[left : right + 1] + y_shift
            fig.add_trace(
                go.Scatter(
                    x=xlist,
                    y=ylist,
                    name="cell" + str(cell_num),
                    mode="lines",
                    line=dict(color=cycle[int(cell_num % 10)]),
                )
            )
            fig.update_layout(showlegend=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
        cell_num += 1

    # ----------------------------------------
    # fig.update_xaxes(rangeslider={"visible":True})
    fig.update_layout(width=500, height=800)
    fig.update_layout(plot_bgcolor="white")
    fig.show()

    return


# ---------------------------------------------------
# ---------------------------------------------------

# ---------------------------------------------------
# ---------------------------------------------------
# statictics_cells
# ---------------------------------------------------
# ---------------------------------------------------
def statictics_cells():
    dia_list = []
    v_list = []
    i = 1
    for cell in confirmed_cells:
        sum_sig = 0
        delta = cell[1][1] - cell[1][0]
        for peak in cell[0]:
            result = re.findall(r"\d+", peak[0])
            column_name = peak[0]
            #            print(peak[0])
            row_number = int(result[0])
            co_number = int(result[1])
            id = row_number * 6 - co_number
            #            id = row_number*6
            y_shift = id
            left = peak[1]
            right = peak[3]
            sum_sig += right - left

        sum_sig = int(((sum_sig / delta) ** 0.2) * 20)
        dia_list.append(sum_sig)
        v_list.append(cell[2])
        i += 1

    for pair in confirmed_pairs:
        delta = pair[0][1][1] - pair[0][1][0]
        sum_sig = 0
        for peak in pair[0][0]:
            #        print(peak[0])
            result = re.findall(r"\d+", peak[0])
            #        print(result)
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = row_number * 6 - co_number
            y_shift = id
            sum_sig += right - left
        for peak in pair[1][0]:
            #        print(peak[0])
            result = re.findall(r"\d+", peak[0])
            #        print(result)
            column_name = peak[0]
            left = peak[1]
            right = peak[3]
            row_number = int(result[0])
            co_number = int(result[1])
            id = row_number * 6 - co_number
            y_shift = id
            sum_sig += right - left

        sum_sig = int(((sum_sig / delta) ** 0.2) * 20)
        dia_list.append(sum_sig)
        v_list.append(pair[0][2] * 0.5 + pair[1][2] * 0.5)

        i += 1

    fig = plt.figure()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(dia_list, bins=10, histtype="barstacked", ec="black")
    ax1.set_title("diameter")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(v_list, bins=10, histtype="barstacked", ec="black")
    ax2.set_title("velocity")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(dia_list, v_list)
    ax3.set_title("v vs d")
    plt.show()
    return dia_list


# ---------------------------------------------------
# ---------------------------------------------------


# ---------------------------------------------------
# ---------------------------------------------------
# MAIN program
# ---------------------------------------------------
# ---------------------------------------------------

sig_con()
draw_graph(0)
df2 = copy.deepcopy(df)
confirmed_cells = []
confirmed_pairs = []

print("calculating correlation")
rep_corr()

for upper_row in reversed(range(1, 16)):
    print("row", upper_row)
    peak_row = []
    create_peak_row(upper_row)
    for i in range(4):
        corr_shift(upper_row)
        #        print("corshift", corshift)
        cell_row = []
        create_cell_row(upper_row)
        #        print("cell_row")
        cell_row_new = []
        create_cell_row_new(upper_row)
        #        print("cell_row_new")
        #    draw_cell(cell_row_new)
        create_pairs()
        #        print("pairs")
        verify_pairs()
        #        print("pairs verified")
        #    draw_pairs()

        found_new = cell_confirm()
        print("found_new", found_new)
        if found_new == 0:
            break
        elif check_remaining() == 0:
            draw_df2(upper_row)
            break
        else:
            draw_df2(upper_row)


# print("confirmed_cells", confirmed_cells)
# print("confirmed_pairs", confirmed_pairs)

print("finalized")
draw_graph2()
final_draw()
print("confirmed cell")
draw_confirmed_cells()
print("confirmed pair")
draw_confirmed_pairs()
print("statistics")
print("total cell num", len(confirmed_cells) + len(confirmed_pairs))

dia_list = statictics_cells()
