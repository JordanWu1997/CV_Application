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
# |_|\_\  |_| |_|  Datetime: 2026-01-14 21:29:41             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import (AutoImageProcessor, AutoModel, Dinov2Config,
                          Dinov2Model)

from utils import save_TF_model_to_local


def resize_with_padding(image, target_size=(224, 224), color=(0, 0, 0)):
    """
    Resizes an image to target_size while keeping aspect ratio and padding with a color.
    """
    old_size = image.shape[:2]  # (height, width)
    ratio = min(float(target_size[i]) / old_size[i] for i in range(2))

    # New dimensions after scaling
    new_size = tuple([int(x * ratio) for x in old_size])

    # Resize the image
    resized = cv2.resize(image, (new_size[1], new_size[0]))

    # Calculate padding
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Add border
    new_img = cv2.copyMakeBorder(resized,
                                 top,
                                 bottom,
                                 left,
                                 right,
                                 cv2.BORDER_CONSTANT,
                                 value=color)
    return new_img


def visualize_all_heads(input_tensor,
                        cls_attn,
                        original_image,
                        global_attn_only=False):
    """
    input_tensor: The tensor fed to DINO (1, 3, H, W)
    cls_attn: Attention weights (num_heads, grid_h, grid_w)
    original_image: Image as a numpy array (H, W, 3)
    """
    num_heads = cls_attn.shape[0]

    # 1. Calculate the Global Importance Map (Average across heads)
    global_attn = torch.mean(cls_attn, dim=0).detach().cpu().numpy()

    if global_attn_only:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # 3. Plot Original Image
        axes[0].imshow(original_image)
        axes[0].set_title("Input Image from Processor")
        axes[0].axis('off')
        # 4. Plot Global Importance Overlay
        global_overlay = plot_attention_overlay(original_image, global_attn)
        axes[1].imshow(global_overlay)
        axes[1].set_title("Global Importance Map")
        axes[1].axis('off')
    else:
        # 2. Setup plotting grid (e.g., if 6 heads, we need 8 slots: Image + Global + 6 heads)
        cols = 4
        rows = (num_heads + 2 + 2 + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = axes.flatten()
        # 3. Plot Original Image
        axes[0].imshow(original_image)
        axes[0].set_title("Input Image from Processor")
        # axes[0].axis('off')
        # 4. Plot Global Importance Overlay
        global_overlay = plot_attention_overlay(original_image, global_attn)
        axes[1].imshow(global_overlay)
        axes[1].set_title("Global Importance Map")
        axes[1].axis('off')
        axes[2].axis('off')
        axes[3].axis('off')
        # 5. Plot Individual Heads
        for i in range(num_heads):
            head_attn = cls_attn[i].detach().cpu().numpy()
            head_overlay = plot_attention_overlay(original_image, head_attn)
            axes[i + 2 + 2].imshow(head_overlay)
            axes[i + 2 + 2].set_title(f"Head {i}")
            axes[i + 2 + 2].axis('off')
        # Hide unused subplots
        for j in range(num_heads + 2 + 2, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_attention_overlay(input_image, attention_map, alpha=0.5):
    """
    input_image: Original numpy image (H, W, 3) - your mean-padded image
    attention_map: The 2D attention grid from a single head (e.g., 16x16)
    alpha: Transparency of the heatmap (0.0 to 1.0)
    """
    # 1. Normalize attention map to 0-255
    attn_min, attn_max = attention_map.min(), attention_map.max()
    norm_attn = (attention_map - attn_min) / (attn_max - attn_min + 1e-8)
    norm_attn = (norm_attn * 255).astype(np.uint8)

    # 2. Resize to match the original image size
    # Using INTER_CUBIC for a smoother "heat" look
    upsampled_attn = cv2.resize(norm_attn,
                                (input_image.shape[1], input_image.shape[0]),
                                interpolation=cv2.INTER_CUBIC)

    # 3. Apply a Colormap
    heatmap = cv2.applyColorMap(upsampled_attn, cv2.COLORMAP_JET)

    # 4. Convert BGR (OpenCV) to RGB for Matplotlib if necessary
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    # 5. Blend the images
    overlay = cv2.addWeighted(input_image, 1 - alpha, heatmap, alpha, 0)

    return overlay


def visualize_image_attn(image_rgb, model, processor, global_attn_only=False):

    # 2. Forward pass (Make sure to output_attentions=True)
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 3. Get attentions from the last layer
    # Shape: (batch_size, num_heads, sequence_length, sequence_length)
    last_layer_attentions = outputs.attentions[-1]

    # 4. Extract [CLS] token attention to all other patches
    # We take the first element of the batch, all heads, and the first row (CLS token)
    # Excluding the first column (CLS attending to itself)
    nh = last_layer_attentions.shape[1]  # Number of heads
    cls_attn = last_layer_attentions[0, :, 0, 1:]

    # 5. Reshape to image grid
    # For a 224x224 image and patch size 14, the grid is 16x16
    grid_size = int(cls_attn.shape[-1]**0.5)
    cls_attn = cls_attn.reshape(nh, grid_size, grid_size)

    # Convert your tensor input back to a displayable numpy image
    # (Undo the normalization and mean-padding if you want the "clean" image)
    img_for_display = inputs['pixel_values'][0].permute(1, 2, 0).cpu().numpy()
    # Rescale to 0-255
    img_for_display = (img_for_display - img_for_display.min()) / (
        img_for_display.max() - img_for_display.min())
    img_for_display = (img_for_display * 255).astype(np.uint8)

    # Execute visualization
    visualize_all_heads(inputs['pixel_values'],
                        cls_attn,
                        img_for_display,
                        global_attn_only=global_attn_only)


if __name__ == '__main__':

    # 1. Load Hugging Face version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_output_root_dir = f'./models'

    # Load config and explicitly enable attentions
    # model_name = "facebook/dinov2-base"
    model_name = "facebook/dinov2-with-registers-base"
    model_output_dir = f'{model_output_root_dir}/{model_name}'
    if os.path.isdir(model_output_dir):
        processor = AutoImageProcessor.from_pretrained(model_output_dir)
        config = Dinov2Config.from_pretrained(model_output_dir)
        config.output_attentions = True
        model = AutoModel.from_pretrained(model_output_dir,
                                          config=config).to(device)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
        config = Dinov2Config.from_pretrained(model_name)
        config.output_attentions = True
        model = AutoModel.from_pretrained(model_name, config=config).to(device)
        save_TF_model_to_local(model, processor, f'{model_output_dir}')

    # Load image
    image_paths = sys.argv[1:]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = resize_with_padding(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visualize_image_attn(image_rgb,
                             model,
                             processor,
                             global_attn_only=False)
