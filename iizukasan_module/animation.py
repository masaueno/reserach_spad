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


def beep(freq, dur=100):
    winsound.Beep(freq, dur)


input_file = "20210531_1_Array2(遮光あり・長方形)1uLmin.csv"
fps = 6.5

windsize = 10
noisesup = 3

time_0 = 3400

gray = [50, 50, 50]
white = [255, 255, 255]
blue = [255, 0, 0]
black = [0, 0, 0]
red = [0, 0, 255]
pink = [200, 200, 255]


p_file = pathlib.Path(input_file)
video_file = p_file.stem + " " + str(fps) + "fps.avi"
anime_file = p_file.stem + ".mp4"
df = pd.read_csv(input_file, nrows=4)
delta = df.iat[3, 1]
df = pd.read_csv(input_file, skiprows=5)
xs = df["No."].iloc[windsize // 2 :] * delta

font = cv2.FONT_HERSHEY_SIMPLEX


def showFrame(frame):
    #    ind = int (( cap.get(0) - time_0 ) / ( delta * 928 ) )
    ind = int((cap.get(1) / fps - time_0 / 1000) / delta)
    #    print(ind, cap.get(1))
    for raw_number in range(16):
        for co_number in range(6):
            column_name = "R" + str(raw_number) + "C" + str(co_number)
            if (ind > windsize // 2) & (ind < 2047 - windsize // 2):
                rad = df[column_name].iloc[ind] * 10
            else:
                rad = -1
            #            print(cap.get(0), ind)
            xl = (points[0][0] * raw_number + points[2][0] * (15 - raw_number)) / 15
            xr = (points[1][0] * raw_number + points[3][0] * (15 - raw_number)) / 15
            x = (xl * (5 - co_number) + xr * co_number) / 5
            x = int(x)
            yl = (points[0][1] * raw_number + points[2][1] * (15 - raw_number)) / 15
            yr = (points[1][1] * raw_number + points[3][1] * (15 - raw_number)) / 15
            y = (yl * (5 - co_number) + yr * co_number) / 5
            y = int(y)

            if (ind > windsize // 2) & (ind < 2047 - windsize // 2):
                spad_color = pink
            else:
                spad_color = [100, 100, 100]

            cv2.circle(frame, (x, y), 5, spad_color, thickness=1)

            if rad > 0.1:
                cv2.circle(frame, (x, y), int(rad / 2.2), (0, 0, 255), thickness=2)
    #                beep(2000, 100)

    if (ind > windsize // 2) & (ind < 2047 - windsize // 2):
        text_color = pink
    else:
        text_color = white

    # time
    cv2.putText(
        frame,
        "{:} [msec]".format(round(count * 1000 / fps)),
        (width + 20, 40),
        font,
        0.5,
        text_color,
        1,
        cv2.LINE_AA,
    )
    # SPAD Start time
    cv2.putText(
        frame, "SPAD start at", (width + 10, 80), font, 0.5, white, 1, cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "{:} [msec]".format(time_0),
        (width + 35, 110),
        font,
        0.5,
        white,
        1,
        cv2.LINE_AA,
    )

    # Moving average

    cv2.putText(
        frame,
        "Moving average size",
        (width + 10, 150),
        font,
        0.5,
        white,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "{:}".format(windsize),
        (width + 35, 180),
        font,
        0.5,
        white,
        1,
        cv2.LINE_AA,
    )

    # Noise suppression

    cv2.putText(
        frame, "Noise suppression", (width + 10, 220), font, 0.5, white, 1, cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "{:}".format(noisesup),
        (width + 35, 250),
        font,
        0.5,
        white,
        1,
        cv2.LINE_AA,
    )

    # fps
    cv2.putText(frame, "fps", (width + 10, 290), font, 0.5, white, 1, cv2.LINE_AA)

    cv2.putText(
        frame, "{:}".format(fps), (width + 35, 320), font, 0.5, white, 1, cv2.LINE_AA
    )

    video.write(frame)
    cv2.imshow("frame", frame)


def comp(x):
    if x > 1:
        return 4 * np.log10(x) + 1
    else:
        return x


# SPAD data noise suppression
for raw_number in range(16):
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
        yc = yc.map(comp)
        df[column_name] = yc.clip(0)


cap = cv2.VideoCapture(video_file)
print(cap.get(3), cap.get(4))
print(cap.get(cv2.CAP_PROP_FPS))

# SPAD 位置合わせ

# データ点格納用
points = np.zeros([4, 2], dtype=int)
global_pt = np.array([0])

# マウスイベント処理(leftimg)
print("click in the order of 左上、右上、左下、右下")


def mouse_event(event, x, y, flags, param):
    # 配列外参照回避
    if global_pt[0] > 3:
        return
    # クリック地点を配列に格納
    if event == cv2.EVENT_LBUTTONUP:
        points[global_pt[0]] = [x, y]  # 格納
        print(x, y)
        cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
        global_pt[0] += 1  # 要素地点を1つ増やす


# 位置取得
ret, frame = cap.read()
frame = cv2.resize(frame, dsize=(int(cap.get(3) / 2.2), int(cap.get(4) / 2.2)))
cv2.imshow("frame", frame)
cv2.setMouseCallback("frame", mouse_event)

while True:
    # 画像の表示
    #    cv2.imshow('frame', frame)
    # 終了処理
    cv2.waitKey(100)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & global_pt[0] == 4:
        break
cap.release()
done = 0

while done == 0:
    # 開始時刻設定
    while 1:
        time_0 = input("開始時刻(ms)>>>")
        if time_0.isdecimal():
            time_0 = int(time_0)
            print(time_0)
            break

    # 動画表示
    cap = cv2.VideoCapture(video_file)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(3) / 2.2)
    height = int(cap.get(4) / 2.2)
    video = cv2.VideoWriter(anime_file, codec, fps, (width + 200, height))
    count = 0

    telop = np.zeros((height, 200, 3), np.uint8)
    telop[:] = tuple((128, 128, 128))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = fgbg.apply(frame)
            frame = cv2.merge([frame, frame, frame])
            frame = cv2.resize(
                frame, dsize=(int(cap.get(3) / 2.2), int(cap.get(4) / 2.2))
            )
            #            frame[np.where((frame == white).all(axis=2))] = blue
            #            frame[np.where((frame == black).all(axis=2))] = white

            images = [frame, telop]
            frame = np.concatenate(images, axis=1)

            showFrame(frame)
            count += 1
            key = cv2.waitKey(300)
            if key == ord("q"):
                done = 1
                break
            elif key != -1:
                break
        else:
            cv2.waitKey(1000)
            done = 1
            break

    video.release()
    cap.release()


cv2.destroyAllWindows()
