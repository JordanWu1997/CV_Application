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

import os
import time

import cv2
from tqdm import tqdm
from uniface import MobileGaze
from uniface.attribute.age_gender import AgeGender
from uniface.attribute.emotion import Emotion
from uniface.constants import (ArcFaceWeights, DDAMFNWeights, GazeWeights,
                               ParsingWeights, RetinaFaceWeights,
                               SphereFaceWeights, YOLOv5FaceWeights)
from uniface.detection import RetinaFace
from uniface.parsing import BiSeNet
from uniface.recognition import ArcFace
from uniface.visualization import draw_detections, draw_gaze, vis_parsing_maps

from utils.face_utils import (align_face_and_embed, detect_face,
                              get_matched_faces,
                              get_target_faces_and_embeddings,
                              match_face_by_similarity, overlay_image_smart)


def run_visualization_pipeline(canvas, vis_threshold=0.6, face_image_size=112):

    # Visualization: Draw matched detections
    draw_detections(image=canvas,
                    bboxes=matched_bboxes,
                    scores=matched_scores,
                    landmarks=matched_landmarks,
                    vis_threshold=vis_threshold)

    # Visualization: Draw non-matched detection
    for bbox in non_matched_bboxes:
        x1, y1, x2, y2 = [int(ele) for ele in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Visualization: Draw matched/non-matched target
    overlay_x, overlay_y, gap = 0, image_height - face_image_size, 6
    target_matched_ids = [
        value for _, value in match_id_dict.items() if value is not None
    ]
    for i, target_aligned_face in enumerate(target_aligned_faces):
        # Add border to image
        if i in target_matched_ids:
            color = [255, 0, 255]
        else:
            color = [255, 255, 255]
        target_aligned_face_vis = cv2.copyMakeBorder(target_aligned_face,
                                                     3, 3, 3, 3, \
                                                     cv2.BORDER_CONSTANT,
                                                     value=color)
        canvas = overlay_image_smart(canvas, target_aligned_face_vis,
                                     int(overlay_x), int(overlay_y))

        # Add text result
        if i in target_matched_ids:
            color = [255, 0, 255]
        else:
            color = [255, 255, 255]
        cv2.putText(canvas, f'{i:d}', (overlay_x + 2, overlay_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(canvas, f'{i:d}', (overlay_x, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        overlay_x = overlay_x + face_image_size + gap

    # Visualization: Draw alignments
    padding = 10
    for i, (face, aligned_face) in \
            enumerate(zip(faces, aligned_faces)):
        bbox = [int(ele) for ele in face.bbox]
        x1, y1, x2, y2 = bbox

        # Aligned faces
        if match_id_dict[i] is not None:
            color = [0, 255, 0]
        else:
            color = [0, 0, 255]

        # Add border to image
        aligned_face_vis = cv2.copyMakeBorder(aligned_face,
                                              3, 3, 3, 3, \
                                              cv2.BORDER_CONSTANT,
                                              value=color)
        aligned_face_h, aligned_face_w, _ = aligned_face_vis.shape
        overlay_x, overlay_y = x2 + padding, y1
        if overlay_x + aligned_face_w > image_width:
            overlay_x = x1 - padding - aligned_face_w
        canvas = overlay_image_smart(canvas, aligned_face_vis, int(overlay_x),
                                     int(overlay_y))

        # Skip unmatched case
        if match_id_dict[i] is None:
            continue

        # Target aligned faces
        target_aligned_face_vis = cv2.copyMakeBorder(
            target_aligned_faces[match_id_dict[i]],
            3, 3, 3, 3, cv2.BORDER_CONSTANT, \
            value=[255, 0, 255])
        target_aligned_face_h, target_aligned_mask_w, _ = target_aligned_face_vis.shape
        overlay_x = x2 + padding
        overlay_y = y1 + aligned_face_h + padding
        if overlay_x + aligned_face_w > image_width:
            overlay_x = x1 - padding - aligned_face_w
        canvas = overlay_image_smart(canvas, target_aligned_face_vis,
                                     int(overlay_x), int(overlay_y))

        # Target result
        overlay_x = x2 + padding
        overlay_y = y1 + aligned_face_h + target_aligned_face_h + padding + gap + 20
        if overlay_x + aligned_face_w > image_width:
            overlay_x = x1 - padding - aligned_face_w
        cv2.putText(canvas,
                    f'{match_id_dict[i]:d}: {max_match_similarities[i]:.2f}',
                    (overlay_x + 2, overlay_y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), 2)
        cv2.putText(canvas,
                    f'{match_id_dict[i]:d}: {max_match_similarities[i]:.2f}',
                    (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 0, 255), 2)

    # Init OSD
    bboxes = [f.bbox for f in faces]
    OSD_text = f'[INFO] Detected Faces: {len(bboxes):d} '
    OSD_text += f'Infer FPS: {1/FS_elapse:.1f} '
    OSD_text += f'Similarity THR: {similarity_threshold:.1f}'

    # Add OSD
    cv2.putText(canvas, OSD_text, (5, 30), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return canvas


if __name__ == '__main__':

    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description="Process images.")
    parser.add_argument('-i',
                        '--input',
                        nargs='+',
                        required=True,
                        help='Input image paths')
    # Add an option for target image paths
    parser.add_argument('-t',
                        '--target',
                        nargs='+',
                        required=True,
                        help='Target image paths')
    # Add an option for output directory (default value provided)
    parser.add_argument('-o',
                        '--output_dir',
                        default='./output',
                        help='Output image directory')
    parser.add_argument('-s',
                        '--similarity_threshold',
                        default=0.3,
                        type=float,
                        help='Similarity threshold')
    args = parser.parse_args()

    # Parse input arguments
    input_image_paths = args.input
    target_image_paths = args.target
    output_image_dir = args.output_dir
    similarity_threshold = args.similarity_threshold
    output_suffix = 'output_FD'

    # # Init model: detector
    # detector = RetinaFace(
    # conf_thresh=0.5,
    # nms_thresh=0.4,
    # )
    # Init model: detector
    detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34,
                          conf_thresh=0.5,
                          nms_thresh=0.4,
                          input_size=(640, 640))

    # Init model: recognizer
    recognizer = ArcFace(model_name=ArcFaceWeights.RESNET)

    # Target images
    target_aligned_faces, target_embeddings = \
        get_target_faces_and_embeddings(target_image_paths, detector, recognizer)

    # Main
    for input_image_path in tqdm(input_image_paths):
        image = cv2.imread(input_image_path)
        image_height, image_width, _ = image.shape

        # Detect face
        FS_start = time.time()
        faces = detect_face(image, detector, confidence_threshold=0.5)

        # Embed face
        align_start = time.time()
        aligned_faces, embeddings = \
            align_face_and_embed(image, recognizer, faces)

        # Match by similarity and get matched faces
        match_start = time.time()
        match_id_dict, max_match_similarities = \
            match_face_by_similarity(embeddings, target_embeddings, similarity_threshold=0.3)
        (matched_bboxes, matched_scores, matched_landmarks), \
        (non_matched_bboxes, non_matched_scores, non_matched_landmarks) = \
            get_matched_faces(faces, match_id_dict)

        FS_elapse = time.time() - FS_start

        # Init canvas
        canvas = image.copy()
        canvas = run_visualization_pipeline(canvas)

        # Save result
        image_name, image_ext = os.path.splitext(
            os.path.basename(input_image_path))
        image_output_path = f'{output_image_dir}/{image_name}_{output_suffix}{image_ext}'
        # Init output dir
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)
        cv2.imwrite(image_output_path, canvas)
