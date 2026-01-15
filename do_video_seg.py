#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2025-04-02 21:02:33             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import argparse
import sys
import time
from random import randint

import cv2
import numpy as np
from ultralytics import SAM, YOLO, FastSAM

from utils.utils import (parse_video_device, put_text_to_canvas, resize_image,
                         toggle_bool_option)


def do_image_segmentation(image, model, mask_only=False):
    # Get image geometry: size
    height, width, _ = image.shape
    # Segmenation
    results = model.predict(image)
    # Init background canvas
    if mask_only:
        result_image = np.zeros_like(image, dtype=np.uint8)
    else:
        result_image = image.copy()
    # Early stop
    if results[0].masks is None:
        return result_image
    # Visualization (segmentation)
    masks = results[0].masks.data.cpu().tolist()
    # Plot segmenation blocks
    for i, mask in enumerate(masks):
        mask = np.array(mask, dtype=bool)
        canvas = np.zeros_like(mask, dtype=np.uint8)
        canvas = np.array([canvas] * 3)
        canvas = np.transpose(canvas, (1, 2, 0))
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        canvas[mask] = color
        canvas = cv2.resize(canvas, (width, height))
        result_image = cv2.addWeighted(result_image, 1, canvas, 0.5, 0)
    return result_image


def do_image_classification(image, model, canvas=None):
    results = model(image)
    if canvas is not None:
        results[0].orig_img = canvas
    result_image = results[0].plot()
    return result_image


def main():

    # Input argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--list-devices',
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
    parser.add_argument('-m',
                        '--mask_only',
                        help='Show segmenation mask for visualization',
                        action='store_true')
    parser.add_argument('-s',
                        '--skip_frame',
                        type=int,
                        default=3,
                        help='Number of frame to skip')
    parser.add_argument('-o',
                        '--option',
                        choices=['seg-cls', 'seg', 'cls'],
                        default='seg-cls',
                        help='Option: combination of seg. and cls.')
    args = parser.parse_args()

    # List available devices
    if args.list_devices:
        devices = get_available_devices(verbose=True)
        sys.exit(f'[INFO] Found devices: {devices}')

    # Get input device
    input_device = parse_video_device(args.input_device, YT_URL=args.YT_URL)

    # Capture URL
    cap = cv2.VideoCapture(input_device)
    if not cap.isOpened():
        print(f'[ERROR] Cannot play {input_device} ...')
        return
    else:
        print(f'[INFO] Start to play {input_device} ...')

    # Init
    show_OSD, mask_only = True, args.mask_only

    # Load segmentation model
    seg_model = FastSAM('./weights/FastSAM-s.pt')

    # Load
    cls_model = YOLO('./weights/yolo11n-cls.pt')

    # Main
    counter, skip_frame, playspeed = 0, args.skip_frame, 1
    while True:

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print('[ERROR] Cannot get image from source ... Retrying ...')
            time.sleep(0.1)
        start = time.time()
        counter += 1

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

        # Get frame property:  FPS
        input_FPS = cap.get(cv2.CAP_PROP_FPS)

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
            skip_frame += 1
        if key == ord('-'):
            skip_frame -= 1
            if skip_frame < 1:
                print('[WARNING] Reached minimal skip_frame: 1')
                skip_frame = 1
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
        # Toggle mask_only option
        if key == ord('m'):
            mask_only = toggle_bool_option(mask_only)
            print(f'[INFO] Mask_only option toggled. Mask_only: {mask_only}')

        # Perform object detection on an image
        if counter % skip_frame == 0 and counter > skip_frame:
            if args.option == 'seg-cls':
                result_image = do_image_segmentation(frame,
                                                     seg_model,
                                                     mask_only=mask_only)
                result_image = do_image_classification(frame,
                                                       cls_model,
                                                       canvas=result_image)
            elif args.option == 'seg':
                result_image = do_image_segmentation(frame,
                                                     seg_model,
                                                     mask_only=mask_only)
            elif args.option == 'cls':
                result_image = do_image_classification(frame, cls_model)
            else:
                print(
                    f'[WARNINGK] Invalid option: {args.option}. Ignore it ...')
            # Infer FPS
            infer_FPS = 1 / (time.time() - start)

        # Use previous result when object detection is ignored at current frame
        try:
            canvas = result_image.copy()
        except UnboundLocalError:
            canvas = frame

        # Infer FPS
        try:
            infer_FPS = infer_FPS
        except UnboundLocalError:
            infer_FPS = -1

        # Add OSD
        OSD_text = f'[{args.option}] '
        if mask_only:
            OSD_text += '[M] '
        OSD_text += f'Input FPS: {input_FPS:.1f}, '
        OSD_text += f'Infer FPS: {infer_FPS:.1f}, '
        OSD_text += f'Playspeed: {playspeed:.2f}, '
        OSD_text += f'Infer every {skip_frame:d} frame'
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
        cv2.imshow(f"FastSAM-s: {input_device}", canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
