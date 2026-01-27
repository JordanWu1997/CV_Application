#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8
r"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2026-01-27 22:42:25             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import sys

import cv2
import yaml
from tqdm import tqdm
from ultralytics import YOLO

if __name__ == '__main__':

    # Load the model
    model = YOLO("./weights/yolo11n.pt")

    input_video_paths = sys.argv[1:]
    for input_video_path in input_video_paths:

        # Load video
        cap = cv2.VideoCapture(input_video_path)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Tracker
        tracker_yaml = './cfg/botsort.yaml'
        with open(tracker_yaml, 'r') as input_yaml:
            tracker_config = yaml.load(input_yaml, Loader=yaml.FullLoader)
        print(tracker_config)

        # Main
        progress_bar = tqdm(total=total_frame)
        while cap.isOpened():
            progress_bar.update(1)

            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame,
                                  persist=True,
                                  verbose=False,
                                  classes=[0],
                                  conf=0.5,
                                  tracker=tracker_yaml)

            canvas = results[0].plot()

            cv2.imshow('Bot-sort Tracking w/ ReID ', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        progress_bar.close()
        cap.release()
        cv2.destoryAllWindows()
