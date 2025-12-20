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
import os
import sys
import time

import cv2
from ultralytics import YOLO

from utils import (get_available_devices, parse_video_device,
                   put_text_to_canvas, resize_image, toggle_bool_option)


def main():
    """  """

    # Model Dict
    model_dict = {
        0: './weights/yolo11n.pt',
        1: './weights/yolov8s-worldv2.pt',
        2: './weights/yolo11n-seg.pt',
        3: './weights/yolo11n-pose.pt',
        4: './weights/yolo11n-obb.pt',
    }

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
    parser.add_argument('-f',
                        '--start_frame',
                        default=0,
                        type=int,
                        help='Frame to start')
    parser.add_argument('-m',
                        '--mode',
                        choices=[i for i in range(5)],
                        default=0,
                        type=int,
                        help=f'{model_dict}')
    parser.add_argument('-t',
                        '--enable_tracking',
                        action='store_true',
                        help='Enable tracking for object detection')
    parser.add_argument('-s',
                        '--skip_frame',
                        type=int,
                        default=3,
                        help='Number of frame to skip')
    parser.add_argument(
        '--classes',
        nargs='+',
        default=None,
        help='Specify classes of objects to detect [mode 1 only]')
    args = parser.parse_args()

    # List available devices
    if args.list_devices:
        devices = get_available_devices(verbose=True)
        sys.exit(f'[INFO] Found devices: {devices}')

    # Init
    show_OSD = True

    # Load model
    model_weight = model_dict[args.mode]
    if not os.path.isfile(model_weight):
        sys.exit(f'[ERROR] Model weight: {model_weight} not found')
    model = YOLO(model_weight)
    if args.classes is not None:
        if args.mode == 1:
            model.set_classes(args.classes)
        else:
            print(
                '[WARNING] Only mode 1 supports classes specification. Ignore it ...'
            )

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

        # Perform object detection on an image
        if counter % skip_frame == 0 and counter > skip_frame:
            # Object Detection w/ or w/o tracking
            if args.enable_tracking:
                results = model.track(frame, persist=True)
            else:
                results = model(frame)
            # Get bboxes, clss, track_ids, elapse
            boxes, elapse = results[0].boxes, results[0].speed
            obj_names = results[0].names
            if boxes is not None:
                bboxes = boxes.xywh.cpu().tolist()
                clss = boxes.cls.int().cpu().tolist()
                if boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
            # Visualize the results on the frame
            canvas = results[0].plot()
            # Infer FPS
            infer_FPS = 1 / (time.time() - start)

        # Use previous result when object detection is ignored at current frame
        else:
            try:
                results[0].orig_img = frame
                canvas = results[0].plot()
            except UnboundLocalError:
                canvas = frame

        # infer_FPS
        try:
            infer_FPS = infer_FPS
        except UnboundLocalError:
            infer_FPS = -1

        # Add OSD
        OSD_text = f'Input FPS: {input_FPS:.1f}, '
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
        cv2.imshow(f"YOLOv11: {input_device}", canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
