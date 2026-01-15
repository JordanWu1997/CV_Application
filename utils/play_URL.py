#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2024-12-22 16:11:23             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import argparse
import time

import cv2

from utils import parse_video_device


def play_URL(input_URL: str, YT_URL: bool = False) -> None:
    """ Play a video from the provided URL. If YT_URL is True, it will convert the input URL to a YouTube URL.

    Args:
        input_URL (str): The URL of the video to be played.
        YT_URL (bool, optional): Whether to convert the input URL to a YouTube URL. Defaults to False.

    Returns:
        None
    """

    # Convert YT URL
    input_URL = parse_video_device(input_URL, YT_URL=YT_URL)

    # Capture URL
    cap = cv2.VideoCapture(input_URL)
    if not cap.isOpened():
        print(f'[ERROR] Cannot play {input_URL} ...')
        return
    else:
        print(f'[INFO] Start to play {input_URL} ...')

    # Main
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print('[ERROR] Cannot get image from source ... Retrying ...')
            time.sleep(0.1)

        # Refresh every 1 milisecond and detect pressed key
        key = cv2.waitKey(1)

        # Break loop when q or Esc is pressed
        if key == ord('q') or key == 27:
            break

        canvas = frame

        # Display
        cv2.imshow(f'Playing: {input_URL}', canvas)

    cap.release()
    cv2.destroyAllWindows()


def main():

    # Input
    parser = argparse.ArgumentParser()
    parser.add_argument('input_URL', type=str, help='Input URL to play')
    parser.add_argument('-y',
                        '--YT_URL',
                        help='If input URL is youtube URL',
                        action='store_true')
    args = parser.parse_args()

    # Main
    play_URL(args.input_URL, YT_URL=args.YT_URL)


if __name__ == '__main__':
    main()
