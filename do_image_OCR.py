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
import os

import cv2
import numpy as np
from rapidocr import (EngineType, LangDet, LangRec, ModelType, OCRVersion,
                      RapidOCR)
from tqdm import tqdm

# from utils.utils import (get_available_devices, parse_video_device,
# put_text_to_canvas, resize_image, toggle_bool_option)

logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)


def main():
    """  """

    # Input argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_image_paths',
                        required=True,
                        type=str,
                        nargs='+',
                        help='Input image paths')
    parser.add_argument('-o',
                        '--output_dir',
                        default='./output',
                        type=str,
                        help='Input image paths')
    parser.add_argument('-c',
                        '--OCR_in_char',
                        action='store_true',
                        help='OCR in character-level')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    args = parser.parse_args()

    # Init OCR Model
    # -- Params: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/parameters/#global
    ocr_model = RapidOCR(
        params={
            'Global.return_word_box': args.OCR_in_char,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CH,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        })

    # Init output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Main
    for input_image_path in tqdm(args.input_image_paths):
        # Load image
        image = cv2.imread(input_image_path)
        # Run inference
        ocr_result = ocr_model(image)
        # Print OCR result
        if args.verbose:
            print(f'[INFO] Image: {input_image_path}')
            print(f'[INFO] OCR: {ocr_result.txts}')
        # Visualization
        canvas = ocr_result.vis()
        # Save result
        image_name, ext = os.path.splitext(os.path.basename(input_image_path))
        suffix = 'OCR'
        if args.OCR_in_char:
            suffix = 'OCR_char'
        cv2.imwrite(f'{args.output_dir}/{image_name}_{suffix}{ext}', canvas)


if __name__ == '__main__':
    main()
