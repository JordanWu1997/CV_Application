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
# |_|\_\  |_| |_|  Datetime: 2025-11-30 20:21:54             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from uniface import ArcFace, MobileFace, SphereFace
from uniface.constants import (ArcFaceWeights, MobileFaceWeights,
                               RetinaFaceWeights, SphereFaceWeights)
from uniface.detection import RetinaFace
from uniface.face_utils import face_alignment
from uniface.visualization import draw_detections


def visualize_FD_result(original_images,
                        detection_images,
                        aligned_images,
                        output_image_path='./output.png',
                        layout='horizontal'):

    # Init output
    if not os.path.isdir(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))

    if layout == 'horizontal':
        fig, axes = plt.subplots(3, len(original_images), figsize=(15, 10))
        row_titles = ["Original", "Detection", "Aligned"]
        for row, images in enumerate(
            [original_images, detection_images, aligned_images]):
            for col, img in enumerate(images):
                axes[row, col].imshow(img)
                axes[row, col].axis("off")
                if col == 0:
                    axes[row, col].set_title(row_titles[row],
                                             fontsize=12,
                                             loc="left")
    else:
        fig, axes = plt.subplots(len(original_images), 3, figsize=(10, 20))
        row_titles = ["Original", "Detection", "Aligned"]
        for row, images in enumerate(
            [original_images, detection_images, aligned_images]):
            for col, img in enumerate(images):
                axes[col, row].imshow(img)
                axes[col, row].axis("off")
                if col == 0:
                    axes[col, row].set_title(row_titles[row],
                                             fontsize=12,
                                             loc="left")

    plt.tight_layout()
    plt.savefig(output_image_path)


def visualize_similarity(aligned_images_1,
                         aligned_images_2,
                         embeddings_1,
                         embeddings_2,
                         threshold=0.4,
                         output_image_path='./output.png'):

    # Init output
    if not os.path.isdir(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))

    from uniface import compute_similarity

    # New figure
    fig, axes = plt.subplots(len(aligned_images) + 1,
                             len(aligned_images) + 1,
                             figsize=(20, 20))

    # Origin
    axes[0][-1].axis('off')
    axes[0][-1].set_title(f'THR: {threshold:.1%}', fontsize=20, color='green')

    # Y-axis
    for i, aligned_image in enumerate(aligned_images):
        i += 1
        axes[i][0].axis('off')
        axes[i][-1].imshow(aligned_image)
        axes[i][-1].axis('off')

    # X-axis
    for j, aligned_image in enumerate(aligned_images):
        axes[0][j].imshow(aligned_image)
        axes[0][j].axis('off')

    for i, (embedding1, image1) in \
            enumerate(zip(embeddings_1, aligned_images_1)):
        i += 1
        for j, (embedding2, image2) in \
                enumerate(zip(embeddings_2, aligned_images_2)):
            j += 1

            if i > j:
                axes[i][j].axis('off')
                continue

            # Calculate similarity
            similarity = compute_similarity(embedding1, embedding2)

            # Filter result with threshold
            color = 'red'
            if similarity > threshold:
                color = 'green'

            # Concat images
            height, width, _ = image1.shape
            resized1 = cv2.resize(image1, (int(width / 2), int(height / 2)))
            height, width, _ = image2.shape
            resized2 = cv2.resize(image2, (int(width / 2), int(height / 2)))
            image = cv2.hconcat([resized1, resized2])

            # Show image
            axes[i][j - 1].imshow(image)
            axes[i][j - 1].set_title(f'{similarity:.1%}',
                                     fontsize=25,
                                     color=color)
            axes[i][j - 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_image_path)


def run_similarirty_comparison(recognizer,
                               faces_in_images,
                               original_images,
                               output_image_path='output.png'):

    # Embed and align
    embeddings, aligned_images = [], []
    for faces, image in zip(faces_in_images, original_images):
        for face in faces:
            # Get landmarks
            landmarks = np.array(face.landmarks)
            # Align
            aligned_image, _ = face_alignment(image, landmarks, image_size=112)
            aligned_images.append(aligned_image)
            # Embed
            embedding = recognizer.get_normalized_embedding(image, landmarks)
            embeddings.append(embedding)

    visualize_similarity(aligned_images,
                         aligned_images,
                         embeddings,
                         embeddings,
                         output_image_path=output_image_path)


if __name__ == '__main__':

    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description="Process frames.")
    parser.add_argument('-i',
                        '--input',
                        nargs='+',
                        required=True,
                        help='Input image paths')
    parser.add_argument('-o',
                        '--output_dir',
                        default='./output',
                        help='Output image directory')

    args = parser.parse_args()

    # Input arguments
    output_image_dir = args.output_dir

    # Init detector
    detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34,
                          conf_thresh=0.5,
                          nms_thresh=0.4,
                          input_size=(640, 640))

    # Input images
    image_paths = []
    for input_path in args.input:
        if os.path.isdir(input_path):
            image_paths.extend(glob.glob(f'{input_path}/*.png'))
        elif os.path.isfile(input_path):
            image_paths.append(input_path)

    # Main
    original_images, detection_images, aligned_images = [], [], []
    faces_in_images, original_images_with_face = [], []
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)

        # Detect faces
        faces = detector.detect(image)
        if not faces:
            print(f"No faces detected in {image_path}")
            continue
        faces_in_images.append(faces)
        original_images_with_face.append(cv2.cvtColor(image,
                                                      cv2.COLOR_BGR2RGB))

        # Align detected face (returns aligned image and inverse transform matrix)
        for face in faces:
            landmarks = np.array(face.landmarks)
            aligned_image, _ = face_alignment(image, landmarks, image_size=112)

            # Draw detections
            detection_image = image.copy()
            draw_detections(image=detection_image,
                            bboxes=[face.bbox],
                            scores=[face.confidence],
                            landmarks=[face.landmarks],
                            vis_threshold=0.6)

            # Convert BGR to RGB for visualization
            original_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            detection_images.append(
                cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB))
            aligned_images.append(
                cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))

    # Visualization
    visualize_FD_result(
        original_images,
        detection_images,
        aligned_images,
        layout='vertical',
        output_image_path=f'{output_image_dir}/output_FD_FA_vertical.png')

    # Visualization
    visualize_FD_result(
        original_images,
        detection_images,
        aligned_images,
        layout='horizontal',
        output_image_path=f'{output_image_dir}/output_FD_FA_horizontal.png')

    # Initialize models
    recognizer = ArcFace()
    # Similarity comparison pipeline
    run_similarirty_comparison(
        recognizer,
        faces_in_images,
        original_images_with_face,
        output_image_path=
        f'{output_image_dir}/output_similarity_ArcFace_MNET.png')

    recognizer = ArcFace(model_name=ArcFaceWeights.RESNET)
    run_similarirty_comparison(
        recognizer,
        faces_in_images,
        original_images_with_face,
        output_image_path=
        f'{output_image_dir}/output_similarity_ArcFace_RESNET.png')

    recognizer = MobileFace(model_name=MobileFaceWeights.MNET_V3_LARGE)
    run_similarirty_comparison(
        recognizer,
        faces_in_images,
        original_images_with_face,
        output_image_path=
        f'{output_image_dir}/output_similarit_MobileFace_MNET_V3_LARGE.png')

    recognizer = SphereFace(model_name=SphereFaceWeights.SPHERE20)
    run_similarirty_comparison(
        recognizer,
        faces_in_images,
        original_images_with_face,
        output_image_path=
        f'{output_image_dir}/output_similarit_SphereFace_SPHERE36.png')
