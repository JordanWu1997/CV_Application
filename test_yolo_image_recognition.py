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

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def main():

    # Model Dict
    model_dict = {
        0: None,
        1: './weights/yolo11n.pt',
        2: './weights/yolov8s-worldv2.pt',
        3: './weights/yolo11n-cls.pt',
        4: './weights/yolo11n-seg.pt',
        5: './weights/yolo11n-pose.pt',
    }

    # Input argument
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image',
                        default=None,
                        type=str,
                        help='Input image')
    parser.add_argument('-m',
                        '--mode',
                        choices=[i for i in range(6)],
                        default=0,
                        type=int,
                        help=f'{model_dict}')
    parser.add_argument('-c',
                        '--color_hist',
                        action='store_true',
                        help='Show color histogram')
    args = parser.parse_args()

    # Read the image using OpenCV
    if not os.path.isfile(args.input_image):
        sys.exit(f'[ERROR] Image: {args.input_image} not found')
    image = cv2.imread(args.input_image)

    # Load model
    model_weight = model_dict[args.mode]
    if model_weight is not None:
        if not os.path.isfile(model_weight):
            sys.exit(f'[ERROR] Model weight: {model_weight} not found')
        model = YOLO(model_weight)

    # YOLO
    if model_weight is not None:
        results = model(image)
        canvas = results[0].plot()
    else:
        canvas = image

    # Convert BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 10))
    if args.color_hist:
        gs = gridspec.GridSpec(2, 1, height_ratios=[7, 3])

    # Display the image in the first subplot
    if args.color_hist:
        ax1 = plt.subplot(gs[0])
    else:
        ax1 = plt.subplot()
    ax1.imshow(canvas_rgb)
    ax1.axis('off')
    if model_weight is not None:
        ax1.set_title(f'Infer: {model_weight}')

    # Histograms
    if args.color_hist:
        ax2 = plt.subplot(gs[1])

        # Calculate histograms for each color channel
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax2.plot(hist,
                     color=color,
                     alpha=0.8,
                     label=f'{color.upper()} Channel')

        # Customize histogram subplot
        ax2.set_title('Color Channel Histograms')
        ax2.set_xlabel('Pixel Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.2)

    # Title
    plt.suptitle(f'Input Image: {args.input_image}')

    # Compact layout
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()
