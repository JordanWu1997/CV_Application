#!/usr/bin/env python3

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def save_TF_model_to_local(model, processor, output_dir):
    """  """

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def generate_query_color_map(query_cfg):
    """Generates a mapping of {'query_text': (B, G, R)}."""
    queries = list(query_cfg.keys())
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(queries), 3), dtype=np.uint8)

    # Create dictionary: { text: (B, G, R) }
    color_map = {
        text: tuple(map(int, colors[i]))
        for i, text in enumerate(queries)
    }
    return color_map


def do_image_DINO(image_path, processor, model, query_config, device='cpu'):

    # Parse input image
    if isinstance(image_path, np.ndarray):
        image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
    else:
        image_pil = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Input texts
    labels = list(query_config.keys())
    text = ' '.join(labels)

    # Preprocess inputs
    inputs = processor(text=text, images=image_pil,
                       return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.Tensor([image_pil.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=target_sizes)[0]
    #print(labels, results['text_labels'], results['labels'])

    filtered_results = []
    for score, label_id, box in zip(results["scores"], results["labels"],
                                    results["boxes"]):
        # Get label
        label_text = labels[labels.index(f'{label_id}.')]
        conf_thresh, area_thresh = query_config[label_text]

        # Calculate Box Area
        box = box.tolist()
        area = (box[2] - box[0]) * (box[3] - box[1])

        # 4. Apply your custom filters
        if score >= conf_thresh and area >= area_thresh:
            filtered_results.append({
                "label": label_text,
                "score": round(score.item(), 3),
                "box": [round(b, 2) for b in box],
                "area": round(area, 2)
            })

    return filtered_results


def visualize_OWL_result(image_path, OWL_results, query_color_map):

    # --- Visual Style Settings ---
    line_thickness = 2  # Thicker bounding box
    font_scale = 0.8  # Larger text
    text_thickness = 2  # Bolder text
    label_offset = 10  # Padding for the text box

    # Read input image
    if isinstance(image_path, np.ndarray):
        image = image_path
    else:
        image = cv2.imread(image_path)

    # Visualization
    for result in OWL_results:
        label = result['label']
        score = result['score']
        x1, y1, x2, y2 = map(int, result['box'])

        # Pull color directly from the pre-generated map
        color = query_color_map.get(label, (255, 255, 255))

        # Draw Box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # Label text
        caption = f"{label}: {score:.2f}"
        (w, h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale, text_thickness)

        # Draw Label
        label_ymin = max(y1, h + label_offset)
        cv2.rectangle(image, (x1, label_ymin - h - label_offset),
                      (x1 + w, label_ymin + baseline), color, -1)
        cv2.putText(image,
                    caption, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255),
                    text_thickness,
                    lineType=cv2.LINE_AA)

    return image


if __name__ == '__main__':

    #model_id = "IDEA-Research/grounding-dino-tiny"
    model_id = "IDEA-Research/grounding-dino-base"

    # Local setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_dir = "./models/"  # Your specific directory

    # Load model
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=local_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, cache_dir=local_dir).to(device)

    # For grounded DINO query
    # -- a/an is required in the beginning
    # -- a preiod is required in the end
    query_config = {
        "a cat.": (0.1, -1),
        "a dog.": (0.1, -1),
        "a remote control.": (0.1, -1),
        "a giraffe.": (0.1, -1),
        "a bike.": (0.1, -1),
    }
    query_color_map = generate_query_color_map(query_config)

    input_image_paths = sys.argv[1:]
    for input_image_path in input_image_paths:
        image = cv2.imread(input_image_path)
        DINO_results = do_image_DINO(input_image_path,
                                     processor,
                                     model,
                                     query_config,
                                     device=device)
        canvas = visualize_OWL_result(image, DINO_results, query_color_map)

        # Live display
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        plt.imshow(canvas)
        plt.title(input_image_path)
        plt.show()
