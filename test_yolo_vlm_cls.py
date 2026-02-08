#!/usr/bin/env python3

import base64
import json
import sys

import cv2
import requests
import torch
from ultralytics import YOLO


def get_vlm_prediction(crop, ollama_url, model_name, prompt, verbose=False):
    """Encodes crop to base64 and queries local Ollama server."""

    # 1. Encode image to base64
    _, buffer = cv2.imencode('.jpg', crop)
    b64_string = base64.b64encode(buffer).decode('utf-8')

    # 2. Prepare payload
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "images": [b64_string]
    }

    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        result = response.json().get("response", "").strip()

        if verbose:
            print(f'[INFO] Prompt: {prompt}')
            print(f'[INFO] {model_name}: {result}')
            print()

        # Extract 0 (positive) or 1 (negative) from the response
        return 0 if "yes" in result.lower() else 1

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return 1  # Default to negative on error


if __name__ == '__main__':

    # Input arguments
    input_image_paths = sys.argv[1:]

    # Ultralytics YOLO
    yolo_model = YOLO('./weights/yolo11x.pt')

    # Ollama VLM as classifier
    ollama_url = "http://localhost:11434/api/generate"
    model_name = "gemma3:4b"
    VLM_prompt = "Is the person in this image female? Answer only yes or no."
    VLM_class_names = {0: "Female", 1: "Not-Female"}

    # Main
    for input_image_path in input_image_paths:
        image = cv2.imread(input_image_path)

        # Only process 'person' class (index 0 in COCO)
        results = yolo_model(image, classes=0, conf=0.5, verbose=False)

        # VLM as classifier
        if results[0].boxes is not None:

            # Work on a copy of the tensor to avoid view/read-only issues
            new_data = results[0].boxes.data.clone()

            for i in range(len(new_data)):

                # Extract coordinates
                x1, y1, x2, y2 = new_data[i, :4].int().tolist()

                # Crop and classify
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    VLM_class = get_vlm_prediction(crop,
                                                   ollama_url=ollama_url,
                                                   model_name=model_name,
                                                   prompt=VLM_prompt,
                                                   verbose=False)
                    new_data[i, 5] = VLM_class

            # Update the Results object
            results[0].boxes.data = new_data
            results[0].names = VLM_class_names

        # Save or display
        results[0].show()
