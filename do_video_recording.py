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
# |_|\_\  |_| |_|  Datetime: 2026-06-28 00:54:04             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import argparse
import sys
import time

import cv2
from tqdm import tqdm

from utils.utils import get_available_devices, resize_for_display


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="USB Camera Video Recorder with cliped Saving")

    parser.add_argument('-l',
                        '--list-devices',
                        action='store_true',
                        help='Get available device list')
    parser.add_argument(
        '-i',
        '--input_device',
        type=str,
        default=None,
        help=
        'Camera index (e.g., 0) or device path. Defaults to first available.')
    parser.add_argument(
        '--output_file_pattern',
        type=str,
        default='output_file_{clip:03d}.mp4',
        help='Output file template pattern (must include {clip:03d}).')
    parser.add_argument('--fps',
                        type=int,
                        default=30,
                        help='Target frames per second (default: 30).')
    parser.add_argument(
        '-t',
        '--total_time',
        type=int,
        default=-1,
        help=
        'Total time to record in seconds. Use -1 to record until "q" is pressed.'
    )
    parser.add_argument(
        '-I',
        '--interval',
        type=int,
        default=120,
        help=
        'Interval in seconds to split video into files. Use -1 to save as a single file.'
    )
    parser.add_argument(
        '-r',
        '--resize_for_display_ratio',
        type=float,
        default=1.0,
        help='Multiplier ratio to scale the display window (default: 1.0).')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.list_devices:
        devices = get_available_devices(verbose=True)
        sys.exit(f'[INFO] Found devices: {devices}')

    # Get video device
    input_device = args.input_device
    try:
        input_device = int(input_device) if input_device is not None else None
    except ValueError:
        pass

    if input_device is None:
        try:
            input_device = get_available_devices(number_of_devices=1)[0]
            print(
                f'[INFO] Use first found device as input device: {input_device}'
            )
        except Exception:
            sys.exit('[ERROR] No video devices found automatically.')

    # Open Camera
    cap = cv2.VideoCapture(input_device)
    if not cap.isOpened():
        sys.exit(f'[ERROR] Cannot open input {input_device} ...')

    # --- FORCE MJPEG COMPRESSION TO UNLOCK HIGH RESOLUTION / FPS ---
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # For 60 FPS
    if int(args.fps) == 60:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = args.fps

    print(f"Camera resolution: {frame_width}x{frame_height} at {fps} FPS")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Tracking time and file clips
    clip_index = 0
    start_time = time.time()
    clip_start_time = start_time

    # Define initial file name and writer
    current_output_file = args.output_file_pattern.format(clip=clip_index)
    out = cv2.VideoWriter(current_output_file, fourcc, fps,
                          (frame_width, frame_height))
    print(f"[INFO] Started recording clip {clip_index}: {current_output_file}")

    # Initialize tqdm progress bar if interval tracking is enabled
    pbar = None
    if args.interval != -1:
        pbar = tqdm(total=args.interval,
                    desc=f"clip {clip_index}",
                    unit="s",
                    leave=True)

    print("Recording started... Press 'q' to stop.")

    try:
        while True:
            current_time = time.time()
            elapsed_total = current_time - start_time
            elapsed_clip = current_time - clip_start_time

            # Check for Total Time limit
            if args.total_time != -1 and elapsed_total >= args.total_time:
                print(
                    f"\n[INFO] Reached total target time limit of {args.total_time}s."
                )
                break

            # Check for clip Interval time out
            if args.interval != -1 and elapsed_clip >= args.interval:
                out.release()  # Close current file

                if pbar:
                    pbar.n = args.interval  # Force close visual gap
                    pbar.refresh()
                    pbar.close()  # Clean up old bar instance

                clip_index += 1
                clip_start_time = time.time()  # Reset clip clock
                elapsed_clip = 0.0

                current_output_file = args.output_file_pattern.format(
                    clip=clip_index)
                out = cv2.VideoWriter(current_output_file, fourcc, fps,
                                      (frame_width, frame_height))

                # Start a fresh tqdm bar for the new file clip
                pbar = tqdm(total=args.interval,
                            desc=f"clip {clip_index}",
                            unit="s",
                            leave=True)

            # Update the progress bar display to show current progress down to milliseconds
            if pbar:
                pbar.n = round(elapsed_clip, 1)
                pbar.refresh()

            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("\nError: Failed to grab a frame.")
                break

            # Write the frame into the current clip file
            out.write(frame)

            # Display handling
            height, width, _ = frame.shape
            canvas = resize_for_display(
                frame,
                width=width,
                height=height,
                resize_ratio=args.resize_for_display_ratio)

            cv2.imshow('Recording Stream (Press Q to Exit)', canvas)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User requested stop.")
                break

    finally:
        # Clean up everything safely
        cap.release()
        if out is not None:
            out.release()
        if pbar:
            pbar.close()
        cv2.destroyAllWindows()
        print("[INFO] Recording complete and all files closed.")
