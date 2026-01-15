#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2025-04-06 21:20:46             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output


def read_image(input_image):
    # Check input
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
        return image
    elif isinstance(input_image, np.ndarray):
        return input_image
    else:
        print('[ERROR] Unknown type of {input_image}')
        return


def load_font(
        font_path='./fonts/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf',
        font_size=20):

    # Try to load a font, use default if not provided or fails
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Default font
            font = ImageFont.load_default()
    except Exception:
        print("[Error] Error when loading font, using default")
        font = ImageFont.load_default()
    return font


def get_OCR_in_chars(input_image, lang='chi_tra+eng'):
    # Load image
    image = read_image(input_image)
    # Get character-level data from Tesseract
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_boxes = pytesseract.image_to_boxes(image_rgb,
                                              lang=lang,
                                              config='--oem 1 --psm 3')
    # Post-processing
    height, width, _ = image.shape
    chars, boxes = [], []
    for i, line in enumerate(result_boxes.split('\n')):
        cols = line.split(' ')
        if len(cols) == 1:
            continue
        char = cols[0]
        x1, y1 = int(cols[1]), height - int(cols[4])
        x2, y2 = int(cols[3]), height - int(cols[2])
        chars.append(char)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    result_dict = {'texts': chars, 'boxes': boxes, 'data': None}
    return result_dict


def get_OCR_in_words(input_image, lang='chi_tra+eng'):
    # Load image
    image = read_image(input_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Get OCR result from Tesseract
    word_data = pytesseract.image_to_data(image_rgb,
                                          output_type=Output.DICT,
                                          lang=lang)
    # Post-processing
    result_dict = {
        'texts': [t for t in word_data['text'] if t.strip() != ''],
        'boxes': [(word_data['left'][i], word_data['top'][i],
                   word_data['width'][i], word_data['height'][i])
                  for i in range(len(word_data['text']))
                  if word_data['text'][i].strip() != ''],
        'block_nums': [
            word_data['block_num'][i] for i in range(len(word_data['text']))
            if word_data['text'][i].strip() != ''
        ],
        'line_nums': [
            word_data['line_num'][i] for i in range(len(word_data['text']))
            if word_data['text'][i].strip() != ''
        ],
    }
    return result_dict


def visualize_OCR_result(
        image,
        result_dict,
        font_path='./fonts/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf',
        font_size=20,
        show_fig=True,
        result_in_horizontal=True,
        output=None):

    # Load image
    image = read_image(image)
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    box_draw = ImageDraw.Draw(image)
    font = load_font(font_path=font_path, font_size=font_size)

    # Create empty canvas for visualizing result
    blank = Image.new('RGB', image.size, (255, 255, 255))
    result_draw = ImageDraw.Draw(blank)

    # Plot results
    texts, boxes = result_dict['texts'], result_dict['boxes']
    for text, (x, y, w, h) in zip(texts, boxes):
        x1, y1, x2, y2 = x, y, x + w, y + h
        # Draw rectangle on original image
        box_draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=1)
        result_draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=1)
        result_draw.text(((x1 + x2) // 2, (y1 + y2) // 2),
                         text,
                         fill="black",
                         font=font,
                         anchor="mm")

    # Concatenate original image and result image
    if result_in_horizontal:
        canvas = np.concatenate((image, blank), axis=1)
    else:
        canvas = np.concatenate((image, blank), axis=0)

    # Plot result canvas
    plt.imshow(canvas)
    if show_fig:
        plt.show()
    if output is not None:
        plt.savefig(output, image)
    return canvas


def main():

    import argparse

    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_images',
                        type=str,
                        nargs='+',
                        help='Input images to do OCR')
    parser.add_argument('-l',
                        '--lang',
                        type=str,
                        default='chi_tra+eng',
                        help='Language for OCR')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default=None,
                        help='Result output directory')
    parser.add_argument('-c',
                        '--OCR_in_char',
                        action='store_true',
                        help='OCR in character-level')
    parser.add_argument('-fs',
                        '--font_size',
                        type=int,
                        default=30,
                        help='Fontsize of OCR result')
    args = parser.parse_args()

    # Init output dir
    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

    for image_path in args.input_images:
        # Check if input is a valid file
        if not os.path.isfile(image_path):
            print(f'[ERROR] {image_path} is not a file ...')
            continue
        # Init Output
        output = None
        if args.output_dir is not None:
            image_path_without_ext, image_ext = os.path.splitext(image_path)
            image_name_without_ext = os.path.basename(image_path_without_ext)
            output = f'{args.output_dir}/{image_name_without_ext}_OCR_out{image_ext}'

        # Main
        image = read_image(image_path)
        if args.OCR_in_char:
            result_dict = get_OCR_in_chars(image, lang=args.lang)
        else:
            result_dict = get_OCR_in_words(image, lang=args.lang)

        # Visualization
        visualize_OCR_result(image,
                             result_dict,
                             output=output,
                             result_in_horizontal=True,
                             show_fig=True,
                             font_size=args.font_size)


if __name__ == '__main__':
    main()
