#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import base64
import re
import sys

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils import put_chinese_text_to_canvas


def get_ocr_grounding(image_path,
                      model="deepseek-ocr",
                      url="http://localhost:11434/api/generate"):

    # Encode image in base64
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    # The <|grounding|> tag triggers coordinate output
    payload = {
        "model": model,
        "prompt": "\n<|grounding|>Extract the text with location.",
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    response = requests.post(url, json=payload)
    return response.json().get("response", "")


def visualize_results(response_text,
                      image_path,
                      font_path='./fonts/simfang.ttf'):

    # Load image to get dimensions
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    annotated_img = 255 * np.ones_like(img)
    boxed_original = img.copy()

    # This matches the text inside ref and the 4 numbers inside the nested brackets of det
    pattern = r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det|>"
    matches = re.findall(pattern, response_text)
    for text, x1, y1, x2, y2 in matches:
        if x1 == '':
            continue

        # Scale coordinates (0-1000 -> actual pixels)
        left = int(int(x1) * w / 1000)
        top = int(int(y1) * h / 1000)
        right = int(int(x2) * w / 1000)
        bottom = int(int(y2) * h / 1000)

        # Draw Bounding Box
        cv2.rectangle(
            boxed_original,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            annotated_img,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2,
        )

        # Draw Text Label
        font_size = max(18, 0.75 * (bottom - top))
        annotated_img = put_chinese_text_to_canvas(
            annotated_img,
            text,
            top_left=(left, top),
            font_path=font_path,
            font_size=font_size,
        )

    # Combine origin and result side-by-side
    combined_view = np.hstack((boxed_original, annotated_img))

    import matplotlib.pyplot as plt
    combined_view = cv2.cvtColor(combined_view, cv2.COLOR_BGR2RGB)
    plt.imshow(combined_view)
    plt.show()


if __name__ == '__main__':
    input_image_paths = sys.argv[1:]
    for input_image_path in tqdm(input_image_paths):
        raw_output = get_ocr_grounding(input_image_path)
        print(f"Model Output: {raw_output}")
        visualize_results(raw_output, input_image_path)
