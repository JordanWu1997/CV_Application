#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, DPTForDepthEstimation

from utils import save_TF_model_to_local


def run_depth_estimate(image, model, proceesor):

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare image
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],  # (height, width)
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Return depth map
    output = prediction.cpu().numpy()
    return output


def visualize_depth_result(original_image, predicted_depth):
    """
    Displays the original image and the depth map side-by-side.
    """
    # 1. Convert predicted depth to a 2D numpy array
    # If the output is still a torch tensor, squeeze it and move to CPU
    if isinstance(predicted_depth, torch.Tensor):
        depth_map = predicted_depth.squeeze().cpu().numpy()
    else:
        depth_map = np.array(predicted_depth)

    # 2. Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 3. Plot Original Image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 4. Plot Depth Map
    # Use 'magma' or 'plasma' colormaps for better depth perception
    im = axes[1].imshow(depth_map, cmap="magma")
    axes[1].set_title("DINOv2 Predicted Depth")
    axes[1].axis("off")

    # Add a colorbar to show the scale
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Model output root dir
    model_output_root_dir = f'./models'

    # Check CUDA support
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =======================================================================
    # DINOv2 w/ Depth Head (DPT)
    # =======================================================================
    # Load model
    model_name = "facebook/dpt-dinov2-base-nyu"
    model_output_dir = f'{model_output_root_dir}/{model_name}'
    if os.path.isdir(model_output_dir):
        processor = AutoImageProcessor.from_pretrained(model_output_dir)
        model = DPTForDepthEstimation.from_pretrained(model_output_dir).to(
            device)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name).to(device)
        save_TF_model_to_local(model, processor, f'{model_output_dir}')

    # Load an image
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_paths = sys.argv[1:]

    # Inference
    for image_path in image_paths:
        image = Image.open(image_path)
        depth_map = run_depth_estimate(image, model, processor)
        visualize_depth_result(image, depth_map)
