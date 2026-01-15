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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import (AutoImageProcessor, AutoModel, Dinov2Config,
                          Dinov2Model)

from utils.utils import save_TF_model_to_local


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


def visualize_cls_token_attn_evolution(image_rgb, model, processor):
    """  """

    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs,
                        output_attentions=True,
                        output_hidden_states=True)
        attentions = outputs.attentions

    # Convert your tensor input back to a displayable numpy image
    # (Undo the normalization and mean-padding if you want the "clean" image)
    img_for_display = inputs['pixel_values'][0].permute(1, 2, 0).cpu().numpy()
    # Rescale to 0-255
    img_for_display = (img_for_display - img_for_display.min()) / (
        img_for_display.max() - img_for_display.min())
    img_for_display = (img_for_display * 255).astype(np.uint8)

    # Setup Plotting Grid (3x4 for 12 layers)
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    num_registers = 0  #getattr(model.config, "num_register_tokens", 0)

    # Metadata for reshaping
    h_px, w_px = inputs['pixel_values'].shape[-2:]
    grid_h, grid_w = h_px // model.config.patch_size, w_px // model.config.patch_size

    # Columns: 1 (Mean) + num_heads
    # figsize = (2 * (num_heads + 1), 2 * num_layers)
    figsize = (10, 8)
    fig, axes = plt.subplots(num_layers, num_heads + 1, figsize=figsize)

    for layer_idx in range(num_layers):
        # Get attention for this layer: [heads, tokens, tokens]
        layer_attn = attentions[layer_idx][0]

        # --- Column 0: Mean Attention ---
        mean_attn = layer_attn.mean(dim=0)
        # [CLS] is row 0; patches start after CLS + Registers
        cls_mean = mean_attn[0,
                             1 + num_registers:].reshape(grid_h,
                                                         grid_w).cpu().numpy()

        ax_mean = axes[layer_idx, 0]
        cls_mean = plot_attention_overlay(img_for_display, cls_mean)
        ax_mean.imshow(cls_mean, cmap='magma', interpolation='bicubic')
        ax_mean.axis('off')
        if layer_idx == 0:
            ax_mean.set_title("Mean Attn.", fontsize=10, fontweight='bold')
        # Label rows with Layer Number
        ax_mean.text(-0.2,
                     0.5,
                     f"Layer {layer_idx + 1}",
                     transform=ax_mean.transAxes,
                     va='center',
                     ha='right',
                     fontsize=12,
                     fontweight='bold')

        # --- Columns 1 to N: Individual Heads ---
        for head_idx in range(num_heads):
            head_attn = layer_attn[head_idx]
            cls_head = head_attn[0, 1 + num_registers:].reshape(
                grid_h, grid_w).cpu().numpy()

            cls_head = plot_attention_overlay(img_for_display, cls_head)

            ax_head = axes[layer_idx, head_idx + 1]
            ax_head.imshow(cls_head, cmap='magma', interpolation='bicubic')
            ax_head.axis('off')

            if layer_idx == 0:
                ax_head.set_title(f"Head {head_idx + 1}", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def visualize_image_feature(image_rgb, model, processor):

    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs,
                        output_attentions=True,
                        output_hidden_states=True)

    # Get attentions from the last layer
    # Shape: (batch_size, num_heads, sequence_length, sequence_length)
    last_layer_attentions = outputs.attentions[-1]
    # Extract [CLS] token attention to all other patches
    nh = last_layer_attentions.shape[1]  # Number of heads
    cls_attn = last_layer_attentions[0, :, 0, 1:]
    # For a 224x224 image and patch size 14, the grid is 16x16
    grid_size = int(cls_attn.shape[-1]**0.5)
    cls_attn = cls_attn.reshape(nh, grid_size, grid_size)
    global_attn = torch.mean(cls_attn, dim=0).detach().cpu().numpy()

    # Shape: [1, 1 + num_patches, 768]
    last_hidden_states = outputs.last_hidden_state
    patch_tokens = last_hidden_states[:, 1:, :]
    features = patch_tokens.squeeze(0).cpu().numpy()
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)
    pca_features = (pca_features - pca_features.min()) \
        / (pca_features.max() - pca_features.min())

    height = inputs['pixel_values'].shape[-2]
    width = inputs['pixel_values'].shape[-1]
    grid_h = height // model.config.patch_size
    grid_w = width // model.config.patch_size

    # Reshape to image dimensions
    pca_img = pca_features.reshape(grid_h, grid_w, 3)

    # Tensor
    pca_tensor = torch.from_numpy(pca_img).permute(2, 0, 1).unsqueeze(0)

    # Convert your tensor input back to a displayable numpy image
    # (Undo the normalization and mean-padding if you want the "clean" image)
    img_for_display = inputs['pixel_values'][0].permute(1, 2, 0).cpu().numpy()
    # Rescale to 0-255
    img_for_display = (img_for_display - img_for_display.min()) / (
        img_for_display.max() - img_for_display.min())
    img_for_display = (img_for_display * 255).astype(np.uint8)

    # Upscale to original image size (e.g., 224x224)
    # Using 'bicubic' makes the patch boundaries smooth
    orig_h, orig_w = img_for_display.shape[1], img_for_display.shape[0]
    upscaled_pca = F.interpolate(pca_tensor,
                                 size=(orig_h, orig_w),
                                 mode='bicubic').squeeze(0)
    upscaled_pca = upscaled_pca.permute(1, 2, 0).numpy()

    # Create the 6-panel plot
    fig, axes = plt.subplots(2, 3, figsize=(8, 4))

    # Panel: Original Image
    axes[0][0].imshow(img_for_display)
    axes[0][0].set_title("Input Image from Processor")
    # axes[0][0].axis("off")

    axes[0][1].imshow(img_for_display)
    axes[0][1].imshow(pca_img)
    axes[0][1].set_title("PCA Components (RGB)")
    axes[0][1].axis("off")

    # Plot Global Importance Overlay
    global_overlay = plot_attention_overlay(img_for_display, global_attn)
    axes[0][2].imshow(global_overlay)
    axes[0][2].set_title("Global Importance Map")
    axes[0][2].axis('off')

    # Panel: Red Channel (1st Principal Component)
    # This usually captures the most significant object/background split
    axes[1][0].imshow(upscaled_pca[:, :, 0], cmap='Reds')
    axes[1][0].set_title("PCA Component 1 (Red)")
    axes[1][0].axis("off")

    # Panel: Green Channel (2nd Principal Component)
    axes[1][1].imshow(upscaled_pca[:, :, 1], cmap='Greens')
    axes[1][1].set_title("PCA Component 2 (Green)")
    axes[1][1].axis("off")

    # Panel: Blue Channel (3rd Principal Component)
    axes[1][2].imshow(upscaled_pca[:, :, 2], cmap='Blues')
    axes[1][2].set_title("PCA Component 3 (Blue)")
    axes[1][2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_cls_and_regs(image_rgb, model, processor):

    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs,
                        output_attentions=True,
                        output_hidden_states=True)

    # 2. Metadata
    num_layers = len(attentions)
    num_reg = model.config.num_register_tokens  # Usually 4
    h_px, w_px = inputs['pixel_values'].shape[-2:]
    grid_h, grid_w = h_px // model.config.patch_size, w_px // model.config.patch_size

    # 3. Create Visualization
    # We will show: [Original Image] | [Layer Map] | [Register Activity Bar]
    fig, axes = plt.subplots(num_layers,
                             2,
                             figsize=(10, 2 * num_layers),
                             gridspec_kw={'width_ratios': [3, 1]})

    for i in range(num_layers):
        # Mean across heads
        avg_attn = attentions[i][0].mean(dim=0)  # [Seq, Seq]

        # A. Spatial Map (Patches only)
        # Patches are usually at the end: [CLS (1) + Reg (4) + Patches (256)]
        cls_to_patches = avg_attn[0,
                                  1 + num_reg:].reshape(grid_h,
                                                        grid_w).cpu().numpy()
        ax_map = axes[i, 0]
        im = ax_map.imshow(cls_to_patches,
                           cmap='magma',
                           interpolation='bicubic')
        ax_map.set_ylabel(f"Layer {i+1}",
                          rotation=0,
                          labelpad=30,
                          fontweight='bold')
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        # B. Register Activity (Bar Chart)
        # How much attention is the CLS token giving to the 4 registers?
        reg_weights = avg_attn[0, 1:1 + num_reg].cpu().numpy()

        ax_bar = axes[i, 1]
        bars = ax_bar.bar(range(num_reg), reg_weights, color='skyblue')
        ax_bar.set_ylim(0, max(avg_attn[0].cpu().numpy().max(),
                               0.1))  # Keep scale consistent
        ax_bar.set_xticks(range(num_reg))
        ax_bar.set_xticklabels([f"R{j}" for j in range(num_reg)])
        if i == 0: ax_bar.set_title("Reg Weights")

    plt.tight_layout()
    plt.show()


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
        # Visualization of last attention layer
        visualize_image_attn(image_rgb,
                             model,
                             processor,
                             global_attn_only=True)
        # Visualization of last hidden layer
        visualize_image_feature(image_rgb, model, processor)
        # Visualization of evolution of cls token attention
        visualize_cls_token_attn_evolution(image_rgb, model, processor)
