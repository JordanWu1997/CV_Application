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
# |_|\_\  |_| |_|  Datetime: 2026-01-21 22:56:42             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO

from torchreid.models import build_model
from torchreid.utils import FeatureExtractor, load_pretrained_weights


def detect_and_crop_persons(image_path,
                            yolo_model,
                            output_folder='crops',
                            verbose=False):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Run inference (classes=[0] filters for only 'person')
    results = yolo_model.predict(source=image_path, classes=[0], conf=0.5)

    # Load original image for cropping
    img = cv2.imread(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]

    # Iterate through the results
    save_filepaths = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy(
        )  # Get bounding boxes in [x1, y1, x2, y2]

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # Crop the person region
            crop = img[y1:y2, x1:x2]

            # Save the crop with suffix _1, _2, etc.
            crop_name = f"{base_name}_{i+1}{ext}"
            save_path = os.path.join(output_folder, crop_name)

            cv2.imwrite(save_path, crop)
            save_filepaths.append(save_path)
            if verbose:
                print(f"Saved: {save_path}")

    return save_filepaths


def get_segmented_person(image_path,
                         yolo_model,
                         output_folder='crops',
                         bg_color=(114, 114, 114),
                         verbose=False):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Run inference (classes=0 for person)
    results = yolo_model.predict(source=image_path, classes=0, conf=0.5)

    # Load original image
    if results[0].masks is None:
        print("No person detected.")
        return []

    # Load original image for cropping
    image = cv2.imread(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]

    # Background image
    background = np.full(image.shape, bg_color, dtype=np.uint8)

    save_filepaths = []
    for i, (mask, box) in \
            enumerate(zip(results[0].masks.data, results[0].boxes.xyxy)):

        # 1. Resize and prepare binary mask
        mask = mask.cpu().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        binary_mask = (mask > 0.5).astype(np.uint8)

        # 2. Extract Person (Foreground)
        # Multiply image by mask (mask is 0 or 1)
        foreground = cv2.bitwise_and(image, image, mask=binary_mask)

        # 3. Extract Background area
        # Invert mask to get background area (1 where person is NOT)
        inv_mask = cv2.bitwise_not(binary_mask * 255)
        bg_part = cv2.bitwise_and(background, background, mask=inv_mask)

        # 4. Combine them
        combined = cv2.add(foreground, bg_part)

        # 5. Crop to bounding box
        x1, y1, x2, y2 = map(int, box)
        crop = combined[y1:y2, x1:x2]

        # Save cropped result
        crop_name = f"{base_name}_{i+1}{ext}"
        save_path = os.path.join(output_folder, crop_name)
        cv2.imwrite(save_path, crop)
        save_filepaths.append(save_path)
        if verbose:
            print(f"Saved: {save_path}")

    return save_filepaths


def letterbox(image, new_shape=(112, 112), color=(114, 114, 114)):
    """Resizes image to a square while maintaining aspect ratio using padding."""
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    image = cv2.copyMakeBorder(image,
                               top,
                               bottom,
                               left,
                               right,
                               cv2.BORDER_CONSTANT,
                               value=color)
    return image


def preprocess_image(image_path,
                     target_size=(256, 128),
                     device='cuda',
                     pad_for_embedding=True):

    # Load image
    if isinstance(image_path, np.ndarray):
        image = image_path
    else:
        image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OSNet standard input size
    if pad_for_embedding:
        padded_image = letterbox(image, new_shape=target_size)
        image_tensor = torch.from_numpy(padded_image.transpose(2, 0,
                                                               1)).float()
    else:
        padded_image = letterbox(image, new_shape=target_size)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

    # Normalize (ImageNet stats)
    image_tensor = image_tensor.div(255).sub_(
        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    image_tensor = image_tensor.div_(
        torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    return image_tensor.unsqueeze(0).to(device), padded_image


def visualize_OSNet_AIN_activation_map(image_path,
                                       model,
                                       device='cuda',
                                       target_size=(256, 128),
                                       pad_for_embedding=True):

    # Inference input size
    image_tensor, padded_image = preprocess_image(
        image_path,
        target_size=target_size,
        pad_for_embedding=pad_for_embedding)
    if not pad_for_embedding:
        padded_image = cv2.imread(image_path)
        padded_image = cv2.resize(padded_image,
                                  (target_size[1], target_size[0]))
        padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

    # Inference
    with torch.no_grad():
        x = model.conv1(image_tensor)
        x = model.maxpool(x)
        target_block = model.conv2[0]
        x1 = target_block.conv1(x)

        streams = []
        for i in range(4):
            out = target_block.conv2[i](x1)
            streams.append(target_block.gate(out))

    # Visualization
    titles = [
        'Input',
        '1x1 Branch',
        '3x3 Branch',
        '5x5 Branch',
        '7x7 Branch',
        'Aggregated',
    ]
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))

    aggregated_am = None
    for i in range(6):
        axes[i].axis('off')
        if i == 0:
            axes[i].imshow(padded_image)
            axes[i].set_title(titles[i])
            continue

        # Process stream (index i-1 because 0 is original image)
        feat = streams[i - 1] if i < 5 else None

        # If it's the aggregated map (last plot)
        if i == 5:
            am = aggregated_am / 4
        else:
            # Generate Activation Map for specific stream
            am = torch.pow(feat, 2).sum(dim=1, keepdim=True)
            am = F.interpolate(am,
                               size=target_size,
                               mode='bilinear',
                               align_corners=False)
            am = am.cpu().numpy().squeeze()
            am = (am - am.min()) / (am.max() - am.min() + 1e-12)

            if aggregated_am is None:
                aggregated_am = am.copy()
            else:
                aggregated_am += am

        axes[i].imshow(padded_image)
        axes[i].imshow(am, cmap='jet', alpha=0.45)
        axes[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()


def embed_and_pad(image_path,
                  extractor,
                  target_size=(256, 128),
                  pad_for_embedding=True):
    image_tensor, padded_image = preprocess_image(
        image_path,
        target_size=target_size,
        pad_for_embedding=pad_for_embedding)
    embedding = extractor(padded_image)
    return embedding, padded_image


def visualize_similarity(aligned_images_1,
                         aligned_images_2,
                         embeddings_1,
                         embeddings_2,
                         threshold=0.4,
                         top_right_only=False,
                         output_image_path='./output.png'):

    # Init output
    if not os.path.isdir(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))

    # New figure
    fig, axes = plt.subplots(len(aligned_images_1) + 1,
                             len(aligned_images_2) + 1,
                             figsize=(20, 20))

    # Origin
    axes[0][-1].axis('off')
    axes[0][-1].set_title(f'THR: {threshold:.1%}', fontsize=20, color='green')

    # Y-axis
    for i, aligned_image in enumerate(aligned_images_1):
        i += 1
        axes[i][0].axis('off')
        axes[i][-1].imshow(aligned_image)
        axes[i][-1].axis('off')

    # X-axis
    for j, aligned_image in enumerate(aligned_images_2):
        axes[0][j].imshow(aligned_image)
        axes[0][j].axis('off')

    for i, (embedding1, image1) in \
            enumerate(zip(embeddings_1, aligned_images_1)):
        i += 1
        for j, (embedding2, image2) in \
                enumerate(zip(embeddings_2, aligned_images_2)):
            j += 1

            if i > j and top_right_only:
                axes[i][j].axis('off')
                continue

            # Calculate similarity
            similarity = F.cosine_similarity(embedding1, embedding2).item()

            # Filter result with threshold
            color = 'red'
            if similarity > threshold:
                color = 'green'

            # Concat images
            height, width, _ = image1.shape
            resized1 = cv2.resize(image1, (int(width / 2), int(height / 2)))
            height, width, _ = image2.shape
            resized2 = cv2.resize(image2, (int(width / 2), int(height / 2)))
            image = cv2.hconcat([resized2, resized1])

            # Show image
            axes[i][j - 1].imshow(image)
            axes[i][j - 1].set_title(f'{similarity:.1%}',
                                     fontsize=25,
                                     color=color)
            axes[i][j - 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_image_path)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the YOLO model
    # yolo_model = YOLO('./weights/yolo11n.pt')
    yolo_model = YOLO('./weights/yolo11n-seg.pt')

    # Load Torchreid Model
    model_name = 'osnet_ain_x1_0'
    weights_path = './models/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'

    # OSNet model for visualization
    OSNet_model = build_model(name=model_name, num_classes=0, pretrained=False)
    load_pretrained_weights(OSNet_model, weights_path)
    OSNet_model.to(device).eval()

    # Extractor for image embedding
    extractor = FeatureExtractor(model_name=model_name,
                                 model_path=weights_path,
                                 device=device)

    # Crop image
    save_image_paths = []
    for image_path in sys.argv[1:]:
        save_image_paths += detect_and_crop_persons(image_path, yolo_model)
        # save_image_paths += get_segmented_person(image_path, yolo_model)

    # Visualization for activation map
    for save_image_path in save_image_paths:
        # visualize_OSNet_AIN_activation_map(save_image_path, OSNet_model)
        visualize_OSNet_AIN_activation_map(save_image_path,
                                           OSNet_model,
                                           target_size=(256, 128),
                                           pad_for_embedding=True)

    # Visualization for similarity comparison
    embeddings, padded_images = [], []
    for image_path in save_image_paths:
        embedding, padded_image = embed_and_pad(image_path,
                                                extractor,
                                                target_size=(256, 128),
                                                pad_for_embedding=True)
        embeddings.append(embedding)
        padded_images.append(padded_image)
    visualize_similarity(padded_images,
                         padded_images,
                         embeddings,
                         embeddings,
                         threshold=0.6,
                         top_right_only=True)
