#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
My Simple Camera Function Collections, including

[Image Processing]
1. Snapshot
2. Brightness, Contrast, Noise-suppression, Grayscale
3. Zoom-in/-out, Rotation, Translation
4. Resize for display (for high/low res. camera on low/high res. monitor)

[Application]
1. QR-Code Decoder
2. Barcode Decoder
3. Optical character recognition (OCR) with Tesseract

[TODO]
1. Add perspective transform
2. Add object detection, tracking, foregournd detection, etc
3. Record as video or GIF (also combine audio input)

[References]
-- https://steam.oxxostudio.tw/category/python/ai/opencv-take-picture.html
-- https://steam.oxxostudio.tw/category/python/ai/opencv-keyboard.html
-- https://steam.oxxostudio.tw/category/python/ai/opencv-qrcode-barcode.html
-- https://steam.oxxostudio.tw/category/python/ai/opencv-negative.html
-- https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
-- https://claude.ai
-- https://github.com/madmaze/pytesseract
-- https://github.com/tesseract-ocr/tesseract
-- https://snyk.io/blog/secure-python-url-validation/

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2024-12-17 20:38:49             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import pyperclip
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pyzbar import pyzbar

from utils.utils import (cycle_options, get_available_devices,
                         parse_video_device, put_chinese_text_to_canvas,
                         put_text_to_canvas, resize_image, toggle_bool_option)


def color_adjust(i: int, c: float, b: int) -> np.uint8:
    """Adjusts the color of an image by applying a linear transformation.

    Args:
        i (int): The input value.
        c (float): The coefficient for the linear transformation.
        b (int): The bias term for the linear transformation.

    Returns:
        np.uint8: The adjusted color value as an 8-bit unsigned integer.
    """
    output = i * (c / 100 + 1) - c + b
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    return output


def geometric_transform(image: np.ndarray,
                        zoom: float = 1.0,
                        rotation: float = 0,
                        center: Optional[tuple] = None) -> np.ndarray:
    """ Applies a geometric transformation to an image.

    Args:
        image (np.ndarray): The input image.
        zoom (float): The zoom factor for the transformation. Defaults to 1.0.
        rotation (float): The rotation angle in degrees. Defaults to 0.
        center (Optional[tuple]): The center point of the transformation. Defaults to the center of the image.

    Returns:
        np.ndarray: The transformed image.
    """
    cy, cx = [i / 2
              for i in image.shape[:-1]] if center is None else center[::-1]
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), rotation, zoom)
    result = cv2.warpAffine(image,
                            rotation_matrix,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


def suppress_noise(image, method='Off'):
    """
    Suppress noise from an image using various methods.

    Parameters:
    - image (numpy.ndarray): The input image.
    - method (str): The noise suppression method. Options are 'Off', 'Gaussian', 'Median', and 'Bilateral'.
      Default is 'Off'.

    Returns:
    - numpy.ndarray: The processed image with noise suppressed.
    """
    if method == 'Off':
        return image
    # Gaussian Blur
    elif method == 'Gaussian':
        # Kernel size must be odd (e.g., 3, 5, 7)
        # Larger kernel = more blurring
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred
    # Median Filtering (excellent for salt-and-pepper noise)
    elif method == 'Median':
        # Kernel size determines the neighborhood to consider
        denoised = cv2.medianBlur(image, 5)
        return denoised
    # Bilateral Filtering (preserves edges better)
    elif method == 'Bilateral':
        # Parameters:
        # - Diameter of pixel neighborhood
        # - Sigma color (larger value = farther colors within neighborhood will be mixed)
        # - Sigma space (larger value = farther pixels influence each other)
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        return bilateral
    else:
        print(f'[ERROR] Not a valid method: {method}')
        return image


def grayscale(image, channels=3):
    """
    Convert an image to grayscale.

    Parameters:
    - image: The input image as a NumPy array.
    - channels (int): The number of color channels for the output image. Default is 3.

    Returns:
    - A new image in grayscale format with the specified number of channels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    multi_chan_gray = cv2.merge([gray] * channels)
    return multi_chan_gray


def negative(image: np.ndarray) -> np.ndarray:
    """
    Inverts the given image by subtracting it from 255.

    Parameters:
    image (np.ndarray): The input image as a NumPy array.

    Returns:
    np.ndarray: The inverted image.
    """
    return 255 - image


def decode_qrcode(image, qrcode_detector, verbose=False):
    """ Decodes a QR code from an image using the provided detector.

    Args:
        image (np.ndarray): The input image containing the QR code.
        qrcode_detector: An instance of a QR code detector class.
        verbose (bool): If True, prints debug information about the decoded QR code.

    Returns:
        bool: True if a QR code was successfully decoded and copied to the clipboard, False otherwise.
    """
    data, bbox, rectified = qrcode_detector.detectAndDecode(image)
    if bbox is not None:
        xs = [int(x) for (x, y) in bbox[0]]
        ys = [int(y) for (x, y) in bbox[0]]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        qrcode_bbox = [xmin, ymin, xmax, ymax]
        cv2.rectangle(image, (qrcode_bbox[0], qrcode_bbox[1]),
                      (qrcode_bbox[2], qrcode_bbox[3]), (0, 0, 255), 2)
        # Copy text to clipboard
        if len(data) > 0:
            put_text_to_canvas(image,
                               data,
                               top_left=(qrcode_bbox[0], qrcode_bbox[1]),
                               fg_color=(0, 0, 255),
                               font_scale=0.75,
                               thickness=2)
            pyperclip.copy(data)
            # Verbose
            if verbose:
                print(f'[INFO] QR Code: {data}')
            return True
    return False


def decode_barcode(image, barcode_detector, verbose=False):
    """ Detects and decodes a barcode from an image using the provided detector.

    Args:
        image (numpy.ndarray): The input image containing the barcode.
        barcode_detector: An object that can detect barcodes in the image.
        verbose (bool): If True, prints information about the detected barcode.

    Returns:
        bool: True if a barcode was successfully decoded and copied to the clipboard, False otherwise.
    """
    data, data_type, bbox = barcode_detector.detectAndDecode(image)
    if bbox is not None:
        xs = [int(x) for (x, y) in bbox[0]]
        ys = [int(y) for (x, y) in bbox[0]]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        barcode_bbox = [xmin, ymin, xmax, ymax]
        cv2.rectangle(image, (barcode_bbox[0], barcode_bbox[1]),
                      (barcode_bbox[2], barcode_bbox[3]), (0, 255, 255), 2)
        # Copy text to clipboard
        if len(data) > 0:
            put_text_to_canvas(image,
                               data,
                               top_left=(barcode_bbox[0], barcode_bbox[1]),
                               fg_color=(0, 255, 255),
                               font_scale=0.75,
                               thickness=2)
            pyperclip.copy(data)
            if verbose:
                print(f'[INFO] Barcode: {data}')
            return True
    return False


def decode_with_zbar(image, verbose=False):
    """ Decode QR codes from an image using the pyzbar library.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        verbose (bool, optional): If True, print decoded texts to the console. Defaults to False.

    Returns:
        bool: True if any QR code is found and decoded, False otherwise.
    """
    decoded_result = pyzbar.decode(image)
    if decoded_result != []:
        for decoded in decoded_result:
            cv2.rectangle(image, (decoded.rect.left, decoded.rect.top),
                          (decoded.rect.left + decoded.rect.width,
                           decoded.rect.top + decoded.rect.height),
                          (0, 255, 255), 2)
            # Decode byte to string
            decoded_text = decoded.data.decode('utf-8')
            # Add text
            put_text_to_canvas(image,
                               decoded_text,
                               top_left=(decoded.rect.left, decoded.rect.top),
                               fg_color=(0, 255, 255),
                               font_scale=0.75,
                               thickness=2)
            if verbose:
                print(f'[INFO] Zbar Decoded: {decoded_text}')
            pyperclip.copy(decoded_text)
        return True
    return False


def main():
    """ """

    # Parse input
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
    parser.add_argument('-o',
                        '--output_dir',
                        default='./',
                        help='Snapshot output directory')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Verbosely print info message and show OSD')
    parser.add_argument('-e',
                        '--exit_after_result_returned',
                        action='store_true',
                        help='Exit application right after result is returned')
    parser.add_argument('-z',
                        '--zbar_decoder',
                        action='store_true',
                        help='Enable QR Code/Barcode decoder (zbar)')
    parser.add_argument('-q',
                        '--qrcode_decoder',
                        action='store_true',
                        help='Enable QR Code decoder (OpenCV)')
    parser.add_argument('-b',
                        '--barcode_decoder',
                        action='store_true',
                        help='Enable Barcode decoder (OpenCV)')
    parser.add_argument(
        '-r',
        '--resize_ratio',
        default=1.0,
        type=float,
        help='Resize for high/low resolution camera live display')

    args = parser.parse_args()

    # List available devices
    if args.list_devices:
        devices = get_available_devices(verbose=args.verbose)
        sys.exit(f'[INFO] Found devices: {devices}')

    # Get input device
    input_device = parse_video_device(args.input_device, YT_URL=args.YT_URL)

    # Get video
    cap = cv2.VideoCapture(input_device)
    if not cap.isOpened():
        sys.exit(f'[ERROR] Cannot open input {input_device} ...')

    # QR Code Detector
    qrcode_detector = cv2.QRCodeDetector()

    # Barode Detector
    barcode_detector = cv2.barcode_BarcodeDetector()

    # Init
    alpha, contrast, brightness = 0, 0, 0
    zoom, rotation, center_x_offset, center_y_offset = 1.0, 0, 0, 0
    zoom_step, rotation_step, offset_step = 0.1, 30, 100
    resize_ratio_step = 0.1
    # Flag
    zbar_decoder_on = args.zbar_decoder
    barcode_decoder_on = args.qrcode_decoder
    qrcode_decoder_on = args.barcode_decoder
    exit_after_result_returned = args.exit_after_result_returned
    verbose = args.verbose
    show_OSD = args.verbose
    # Resize for display
    resize_ratio = args.resize_ratio
    # Noise suppression
    noise_suppression_methods = ['Off', 'Gaussian', 'Median', 'Bilateral']
    noise_suppression_method = noise_suppression_methods[0]
    # Grayscale, Negative
    grayscale_on, negative_on = False, False

    # Main
    counter = 0
    while True:

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print('[ERROR] Cannot get image from source ... Retrying ...')
            time.sleep(0.1)
            continue
        counter += 1

        # Get image geometry: size, center
        height, width, _ = frame.shape
        center_x, center_y = int(round(width / 2)), int(round(height / 2))

        # Geometric transform: zoom, rotation, translation
        if not (zoom == 1 and zoom == 0 and center_x_offset == 0
                and center_y_offset == 0):
            center_x += center_x_offset
            center_y += center_y_offset
            frame = geometric_transform(frame,
                                        zoom=zoom,
                                        rotation=rotation,
                                        center=(center_x, center_y))

        # Init per frame
        OSD_text = ''

        # Timer
        start = time.time()

        # Refresh every 1 milisecond and detect pressed key
        key = cv2.waitKey(1)

        # ====================================================================
        # Module: Quit application
        # ====================================================================
        # Break loop when q or Esc is pressed
        if key == ord('q') or key == 27:
            break

        # ====================================================================
        # Module: Zoom / Rotation
        # ====================================================================
        # Dynamic offset step
        dyna_offset_step = offset_step
        if zoom > 1.5:
            dyna_offset_step /= zoom
            dyna_offset_step = int(dyna_offset_step)
            if dyna_offset_step < 10:
                dyna_offset_step = 10
        # Zoom-in/-out y-center (up/down)
        if key == ord('k'):
            center_y_offset -= dyna_offset_step
        if key == ord('j'):
            center_y_offset += dyna_offset_step
        # Zoom-in/-out x-center (left/right)
        if key == ord('h'):
            center_x_offset -= dyna_offset_step
        if key == ord('l'):
            center_x_offset += dyna_offset_step
        # Zoom-in
        if key == ord('='):
            zoom += 0.1
            print(f'[INFO] Zoom: {zoom:.1f}')
        # Zoom-out
        if key == ord('-'):
            zoom -= zoom_step
            if zoom < zoom_step:
                print(f'[Warning] Reached minimal zoom {zoom_step}')
                zoom = zoom_step
            print(f'[INFO] Zoom: {zoom:.1f}')
        # Rotation
        if key == ord('r'):
            rotation -= rotation_step
            if abs(rotation) > 360:
                rotation = 0
            print(f'[INFO] Rotation: {rotation:d}')
        # Restore geometric transform to default
        if key == ord('d'):
            zoom, rotation, center_x_offset, center_y_offset = 1.0, 0, 0, 0
        # Show info on OSD
        OSD_text += f'R: {rotation:d} Z: {zoom:.1f} '
        OSD_text += f'X: {center_x_offset:d} Y: {center_y_offset:d} '

        # ====================================================================
        # Module: Noise suppression
        # ====================================================================
        if key == ord('n'):
            noise_suppression_method = cycle_options(
                noise_suppression_method, noise_suppression_methods)
        frame = suppress_noise(frame, method=noise_suppression_method)
        OSD_text += f'NS: {noise_suppression_method[:3]} '

        # ====================================================================
        # Module: Grayscale
        # ====================================================================
        if key == ord('g'):
            grayscale_on = toggle_bool_option(grayscale_on)
        if grayscale_on:
            frame = grayscale(frame)

        # ====================================================================
        # Module: Negative
        # ====================================================================
        if key == ord('i'):
            negative_on = toggle_bool_option(negative_on)
        if negative_on:
            frame = negative(frame)

        # ====================================================================
        # Module: Adjust color parameters
        # ====================================================================
        # Set contrast (down/up <-> dec/inc)
        if key == 81:
            brightness -= 5
        if key == 83:
            brightness += 5
        # Set brightness (left/right <-> dec/inc)
        if key == 82:
            contrast += 5
        if key == 84:
            contrast -= 5
        # Restore brightness and contrast
        if key == 8:  # Backspace
            brightness, contrast = 0, 0
        # Show brightness/contrast on OSD
        OSD_text += f'B: {brightness:d} C: {contrast:d} '
        # Change image from BGR to BGRA (a for alpha)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        # Adjust contrast and brightness
        frame = color_adjust(frame, contrast, brightness)

        # ====================================================================
        # Module: Take a snapshot
        # ====================================================================
        # Set alpha value to 1 when space is pressed
        if key == 32:  # Spacebar
            alpha = 1
        # Take a snapshot when alpha decreases from 1 to 0
        if alpha != 0:
            # Assign alpha value for visualization and save
            photo = frame.copy()
            white = np.ones_like(frame, dtype='uint8') * 255
            canvas = cv2.addWeighted(white, alpha, photo, 1 - alpha, 0)
            alpha -= 0.1
            if alpha < 0:
                alpha = 0
                current = datetime.now().strftime("%y%m%d%H%M%S%f")[:-3]
                if not os.path.isdir(args.output_dir):
                    os.makedirs(args.output_dir)
                cv2.imwrite(f'{args.output_dir}/{current}.jpg', photo)
            # Add text
            put_text_to_canvas(canvas,
                               'Snapshot Taken',
                               top_left=(10, 70),
                               font_scale=1.5,
                               thickness=3)
        else:
            canvas = frame.copy()

        # ====================================================================
        # Module: Decode QR Code
        # ====================================================================
        if key == ord('c'):
            qrcode_decoder_on = toggle_bool_option(qrcode_decoder_on)
        if qrcode_decoder_on:
            OSD_text += '[CV QR Code Decoder] '
            if decode_qrcode(canvas, qrcode_detector, verbose=verbose):
                if exit_after_result_returned:
                    break

        # ====================================================================
        # Module: Decode barcode
        # ====================================================================
        if key == ord('b'):
            barcode_decoder_on = toggle_bool_option(barcode_decoder_on)
        if barcode_decoder_on:
            OSD_text += '[CV Barcode Decoder] '
            if decode_barcode(canvas, barcode_detector, verbose=verbose):
                if exit_after_result_returned:
                    break

        # ====================================================================
        # Module: Decode QRCode/Barcode with zbar
        # ====================================================================
        if key == ord('z'):
            zbar_decoder_on = toggle_bool_option(zbar_decoder_on)
        if zbar_decoder_on:
            OSD_text += '[Zbar Decoder] '
            if decode_with_zbar(canvas, verbose=verbose):
                if exit_after_result_returned:
                    break

        # ====================================================================
        # Module: Calculate FPS (NOTE: FPS = 1 / elapse)
        # ====================================================================
        FPS = 1 / (time.time() - start)
        OSD_text = f'FPS: {FPS:.1f} {OSD_text}'

        # ====================================================================
        # Module: Miscellaneous Flags
        # ====================================================================
        # Verbose
        if key == ord('v'):
            verbose = toggle_bool_option(verbose)
        if verbose:
            OSD_text = f'[V] {OSD_text}'
        # Exit application after result is returned
        if key == ord('e'):
            exit_after_result_returned \
                = toggle_bool_option(exit_after_result_returned)
        if exit_after_result_returned:
            OSD_text = f'[E] {OSD_text}'

        # ====================================================================
        # Module: Resize for display
        # ====================================================================
        canvas = resize_image(canvas,
                              width=width,
                              height=height,
                              resize_ratio=resize_ratio)
        # Expand
        if key == ord('+'):
            resize_ratio += resize_ratio_step
            print(f'[INFO] Resize ratio: {resize_ratio:.1f}')
        # Shrink
        if key == ord('_'):
            resize_ratio -= resize_ratio_step
            if resize_ratio < resize_ratio_step:
                print(f'[Warning] Reached minimal resize ratio {resize_ratio}')
                resize_ratio = resize_ratio_step
            print(f'[INFO] Resize ratio: {resize_ratio:.1f}')
        # Add resize ratio to OSD
        if round(resize_ratio, 3) != 1.0:
            OSD_text = f'[R: {resize_ratio:.1f}] {OSD_text}'

        # ====================================================================
        # Module: On-Screen Display (OSD)
        # ====================================================================
        if key == 13:  # Enter
            show_OSD = toggle_bool_option(show_OSD)
        if show_OSD:
            put_text_to_canvas(canvas,
                               OSD_text,
                               top_left=(10, 30),
                               font_scale=0.5,
                               fg_color=(0, 255, 0),
                               thickness=1)

        # Display
        cv2.imshow(f'Device: {input_device}', canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
