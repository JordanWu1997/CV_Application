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
# |_|\_\  |_| |_|  Datetime: 2025-11-30 20:20:17             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import glob
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from uniface import MobileGaze
from uniface.attribute.age_gender import AgeGender
from uniface.attribute.emotion import Emotion
from uniface.constants import (DDAMFNWeights, GazeWeights, ParsingWeights,
                               RetinaFaceWeights, YOLOv5FaceWeights)
from uniface.detection import RetinaFace, YOLOv5Face
from uniface.parsing import BiSeNet
from uniface.visualization import draw_detections, draw_gaze, vis_parsing_maps

from utils.face_utils import (align_face, detect_face, estimate_gaze,
                              overlay_image_smart, parse_face,
                              predict_age_gender, predict_emotion)


def run_visualization_pipeline(canvas,
                               vis_threshold=0.6,
                               output_image_width=-1,
                               output_image_height=-1,
                               face_image_size=112,
                               do_face_parsing=True):

    # Get face infos
    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    landmarks = [f.landmarks for f in faces]

    # Draw detections
    draw_detections(image=canvas,
                    bboxes=bboxes,
                    scores=scores,
                    landmarks=landmarks,
                    vis_threshold=vis_threshold)
    # Draw alignments
    padding = 10
    for i, (face, aligned_face) in \
            enumerate(zip(faces, aligned_faces)):
        bbox = [int(ele) for ele in face.bbox]
        x1, y1, x2, y2 = bbox
        # Aligned Faces
        aligned_face_vis = cv2.copyMakeBorder(
            aligned_face,
            3, 3, 3, 3, \
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255])
        aligned_face_h, aligned_face_w, _ = aligned_face_vis.shape
        overlay_x, overlay_y = x2 + padding, y1
        if overlay_x + aligned_face_w > image_width:
            overlay_x = x1 - padding - aligned_face_w
        canvas = overlay_image_smart(canvas, aligned_face_vis, int(overlay_x),
                                     int(overlay_y))

    # Gaze
    for i, (face, aligned_face) in \
            enumerate(zip(faces, aligned_faces)):
        if pitchs[i] is not None and yaws[i] is not None:
            draw_gaze(canvas, face.bbox, pitchs[i], yaws[i])

    # Gender, Age, Emotions
    for i, (face, aligned_face) in \
            enumerate(zip(faces, aligned_faces)):
        bbox = [int(ele) for ele in face.bbox]
        x1, y1, x2, y2 = bbox
        cv2.putText(canvas, f'{gender_strs[i]} {ages[i]} {emotions[i]}',
                    (x1 + 5 + 2, y2 + 20 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 0), 2)
        cv2.putText(canvas, f'{gender_strs[i]} {ages[i]} {emotions[i]}',
                    (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2)

    # Face Masks
    for i, (face, aligned_face) in \
            enumerate(zip(faces, aligned_faces)):
        bbox = [int(ele) for ele in face.bbox]
        x1, y1, x2, y2 = bbox
        if do_face_parsing:
            aligned_face_mask_vis = cv2.copyMakeBorder(
                aligned_face_masks_vis[i],
                3, 3, 3, 3, \
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255])
            overlay_x = x2 + padding
            aligned_face_mask_h, aligned_face_mask_w, _ = aligned_face_mask_vis.shape
            overlay_y = y1 + aligned_face_mask_h + padding
            if overlay_x + aligned_face_mask_w > image_width:
                overlay_x = x1 - padding - aligned_face_mask_w
            canvas = overlay_image_smart(canvas, aligned_face_mask_vis,
                                         int(overlay_x), int(overlay_y))

    # Add OSD
    OSD_text = f'[INFO] Detected Faces: {len(bboxes):d} '
    OSD_text += f'FPS: {1/FD_elapse:.1f} '

    # Add OSD
    cv2.putText(canvas, OSD_text, (5, 30), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Resize
    if output_image_width > 0 and output_image_height > 0:
        canvas = cv2.resize(canvas, (output_image_width, output_image_height))

    return canvas


if __name__ == '__main__':

    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description="Process images.")
    parser.add_argument('-i',
                        '--input',
                        nargs='+',
                        required=True,
                        help='Input video paths')
    parser.add_argument('-o',
                        '--output_dir',
                        default='./output',
                        help='Output video directory')
    parser.add_argument('-p',
                        '--parse_face',
                        action='store_true',
                        help='Face component segmenation')
    args = parser.parse_args()

    # Input arguments
    input_image_paths = args.input
    output_image_dir = args.output_dir
    do_face_parsing = args.parse_face
    output_suffix = 'output_FD'
    add_alignment_visualization = True
    do_face_parsing = True

    # # Init detector
    # detector = RetinaFace(
    # conf_thresh=0.5,
    # nms_thresh=0.4,
    # )
    # Init detector
    detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34,
                          conf_thresh=0.5,
                          nms_thresh=0.4,
                          input_size=(640, 640))
    # # Init detector
    # detector = YOLOv5Face(model_name=YOLOv5FaceWeights.YOLOV5S,
    # conf_thresh=0.6,
    # nms_thresh=0.5)

    # Init Age/Gender detector
    age_gender = AgeGender()

    # Init emotion predictor (requires torch)
    emotion_predictor = Emotion(model_weights=DDAMFNWeights.AFFECNET8,
                                input_size=(112, 112))

    # Init gaze esimator
    gaze_estimator = MobileGaze(model_name=GazeWeights.RESNET18)

    # Init face parsing
    parser = BiSeNet(model_name=ParsingWeights.RESNET34, input_size=(112, 112))

    # Init face anti-spoofing

    # Main
    for input_image_path in tqdm(input_image_paths):
        image = cv2.imread(input_image_path)
        image_height, image_width, _ = image.shape

        # Detect face
        FD_start = time.time()
        faces = detect_face(image, detector, confidence_threshold=0.6)

        # Age, Gender
        age_gender_start = time.time()
        ages, gender_strs = predict_age_gender(image, age_gender, faces)

        # Emotion
        emotion_start = time.time()
        emotions = predict_emotion(image, emotion_predictor, faces)

        # Gaze
        gaze_start = time.time()
        pitchs, yaws = estimate_gaze(image, gaze_estimator, faces)

        # Align detected face (returns aligned image and inverse transform matrix)
        align_start = time.time()
        aligned_faces = align_face(image, faces)

        # Parse aligned face
        parse_start = time.time()
        if do_face_parsing:
            aligned_face_masks, aligned_face_masks_vis = \
                parse_face(image, parser, aligned_faces)

        FD_elapse = time.time() - FD_start

        # Init canvas
        canvas = image.copy()

        # Visualization
        canvas = run_visualization_pipeline(canvas,
                                            do_face_parsing=do_face_parsing)

        # Save result
        image_name, image_ext = os.path.splitext(
            os.path.basename(input_image_path))
        image_output_path = f'{output_image_dir}/{image_name}_{output_suffix}{image_ext}'
        # Init output dir
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)
        cv2.imwrite(image_output_path, canvas)
