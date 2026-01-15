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
import logging
import sys
import time

import cv2
import numpy as np
from rapidocr import (EngineType, LangDet, LangRec, ModelType, OCRVersion,
                      RapidOCR)

from utils.utils import (get_available_devices, parse_video_device,
                         put_text_to_canvas, resize_image, toggle_bool_option)

logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)


def main():
    """  """

    # Input argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-devices',
                        action='store_true',
                        help='Get available device list')
    parser.add_argument('-i',
                        '--input_device',
                        default=None,
                        type=str,
                        help='Input device, file or strearming URL')
    parser.add_argument('-y',
                        '--YT_URL',
                        help='If input URL is youtube URL',
                        action='store_true')
    parser.add_argument('-r',
                        '--resize_ratio',
                        default=1.0,
                        type=float,
                        help='Ratio to resize live display')
    parser.add_argument('-f',
                        '--start_frame',
                        default=0,
                        type=int,
                        help='Frame to start')
    parser.add_argument('-I',
                        '--infer_frame_interval',
                        type=int,
                        default=3,
                        help='Number of frame to skip')
    parser.add_argument('-l',
                        '--lang',
                        type=str,
                        default='chi_tra+eng',
                        help='Language for OCR')
    parser.add_argument('-c',
                        '--OCR_in_char',
                        action='store_true',
                        help='OCR in character-level')
    args = parser.parse_args()

    # List available devices
    if args.list_devices:
        devices = get_available_devices(verbose=True)
        sys.exit(f'[INFO] Found devices: {devices}')

    # Init
    show_OSD = True
    debug = False

    # Get input device
    input_device = parse_video_device(args.input_device, YT_URL=args.YT_URL)

    # Capture URL
    cap = cv2.VideoCapture(input_device)
    if not cap.isOpened():
        print(f'[ERROR] Cannot play {input_device} ...')
        return
    else:
        print(f'[INFO] Start to play {input_device} ...')

    # Get frame property:  FPS
    input_FPS = cap.get(cv2.CAP_PROP_FPS)

    # Jump to frame to start
    if args.start_frame > 0:
        print(f'[INFO] Jump to frame {args.start_frame} ...')
        for _ in range(args.start_frame):
            _, _ = cap.read()

    # Init OCR Model
    # -- Params: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/parameters/#global

    # ocr_model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)
    # ocr_model = RapidOCR(params={'Global.return_word_box': True})
    ocr_model = RapidOCR(
        params={
            'Global.return_word_box': True,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CH,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        })

    # Main
    frame_num, infer_frame_interval, playspeed = 0, args.infer_frame_interval, 1
    while cap.isOpened:

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print('[ERROR] Cannot get image from source ... Retrying ...')
            time.sleep(0.1)
        start = time.time()
        frame_num += 1

        # Speedup playing
        if playspeed > 1:
            for _ in range(int(playspeed) - 1):
                ret, frame = cap.read()
                if not ret:
                    print(
                        '[ERROR] Cannot get image from source ... Retrying ...'
                    )
                    time.sleep(0.1)
            start = time.time()
        elif playspeed < 1 and playspeed > 0:
            FPS = input_FPS * playspeed
            time.sleep(1 / FPS)
            ret, frame = cap.read()
            if not ret:
                print('[ERROR] Cannot get image from source ... Retrying ...')
                time.sleep(0.05)
            start = time.time()

        # Get image geometry: size
        height, width, _ = frame.shape

        # Resize frame
        if round(args.resize_ratio, 3) != 1.0:
            frame = resize_image(frame,
                                 width=width,
                                 height=height,
                                 resize_ratio=args.resize_ratio)

        # Refresh every 1 milisecond and detect pressed key
        key = cv2.waitKey(1)

        # Break loop when q or Esc is pressed
        if key == ord('q') or key == 27:
            break
        # Modify frame to skip
        if key == ord('='):
            infer_frame_interval += 1
        if key == ord('-'):
            infer_frame_interval -= 1
            if infer_frame_interval < 1:
                print('[WARNING] Reached minimal infer_frame_interval: 1')
                infer_frame_interval = 1
        # Fast-forward: 1 sec
        if key == 83:  # Right
            fast_forward_sec = 1
            print(f'[INFO] Fast-forward {fast_forward_sec} secs ...')
            for _ in range(int(input_FPS * fast_forward_sec) - 1):
                _, _ = cap.read()
            continue
        # Fast-forward: 10 sec
        if key == 82:  # Up
            fast_forward_sec = 10
            print(f'[INFO] Fast-forward {fast_forward_sec} secs ...')
            for _ in range(int(input_FPS * fast_forward_sec) - 1):
                _, _ = cap.read()
            continue
        # Speedup playspeed
        if key == ord('s'):
            if playspeed >= 1.0:
                playspeed += 1.0
            else:
                playspeed += 0.25
        # Speeddown playspeed
        if key == ord('a'):
            if playspeed >= 2.0:
                playspeed -= 1.0
            else:
                playspeed -= 0.25
                playspeed = max(playspeed, 0.25)

        # Perform object detection on an image
        if frame_num % infer_frame_interval == 0 \
                and frame_num > infer_frame_interval:
            ocr_result = ocr_model(frame)
            # Infer FPS
            infer_FPS = 1 / (time.time() - start)

        # Use previous result when object detection is ignored at current frame
        canvas = frame.copy()
        try:
            ocr_result.img = canvas
            canvas = ocr_result.vis()
            if canvas is None:
                blank = np.ones_like(frame) * 255
                canvas = cv2.hconcat([frame.copy(), blank])
        except NameError as error:
            if debug:
                print(error)

        # Use previous FPS if no new inference
        try:
            infer_FPS
        except NameError:
            infer_FPS = -1

        # Add OSD
        OSD_text = f'Input FPS: {input_FPS:.1f}, '
        OSD_text += f'Infer FPS: {infer_FPS:.1f}, '
        OSD_text += f'Playspeed: {playspeed:.2f}, '
        OSD_text += f'Infer every {infer_frame_interval:d} frame, '
        if key == 13:  # Enter
            show_OSD = toggle_bool_option(show_OSD)
        if show_OSD:
            put_text_to_canvas(canvas,
                               OSD_text,
                               top_left=(10, 30),
                               font_scale=0.5,
                               fg_color=(0, 255, 0),
                               thickness=1)

        # Display the annotated frame
        cv2.imshow(f"OCR: {input_device}", canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
