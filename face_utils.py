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

import cv2
import numpy as np
from uniface import compute_similarity
from uniface.face_utils import face_alignment
from uniface.visualization import draw_detections, vis_parsing_maps


def overlay_image_smart(bg_img, overlay_img, x_pos, y_pos):
    """
    Overlays image ensuring we don't crash by going out of bounds.
    If the overlay goes off the edge, we clip it.
    """
    h_bg, w_bg = bg_img.shape[0:2]
    h_ov, w_ov = overlay_img.shape[0:2]

    # Calculate valid bounds
    y1, y2 = max(0, y_pos), min(h_bg, y_pos + h_ov)
    x1, x2 = max(0, x_pos), min(w_bg, x_pos + w_ov)

    # Calculate the corresponding slice of the overlay image
    # (This handles cases where the overlay is partially off-screen)
    y1_o = max(0, -y_pos)
    x1_o = max(0, -x_pos)
    y2_o = y1_o + (y2 - y1)
    x2_o = x1_o + (x2 - x1)

    # Do the overlay if dimensions are valid
    if y2 > y1 and x2 > x1:
        bg_img[y1:y2, x1:x2] = overlay_img[y1_o:y2_o, x1_o:x2_o]

    return bg_img


def get_target_faces_and_embeddings(target_image_paths,
                                    detector,
                                    recognizer,
                                    image_size=112):
    target_aligned_faces, target_embeddings = [], []
    for target_image_path in target_image_paths:
        target_image = cv2.imread(target_image_path)
        faces = detector.detect(target_image)
        for face in faces:
            landmarks = np.array(face.landmarks)
            # Align face
            target_aligned_face, _ = face_alignment(target_image,
                                                    landmarks,
                                                    image_size=image_size)
            target_aligned_faces.append(target_aligned_face)
            # Embed faces
            target_embedding = \
                recognizer.get_normalized_embedding(target_image, landmarks)
            target_embeddings.append(target_embedding)
    return target_aligned_faces, target_embeddings


def generate_output_video_writer(input_video_path,
                                 input_video_cap,
                                 output_frame_width=-1,
                                 output_video_dir='',
                                 output_suffix='',
                                 output_video='',
                                 verbose=False):

    FPS = input_video_cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_frame_width = frame_width
    if output_frame_width > 0:
        output_frame_height = int(output_frame_width *
                                  (frame_height / frame_width))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    input_video_name, _ = os.path.splitext(input_video_path)
    output_video_path = f'{input_video_name}_{output_suffix}.mp4'
    if output_video_dir != '':
        # Init output dir
        if not os.path.isdir(output_video_dir):
            os.makedirs(output_video_dir)
        output_video_path = f'{output_video_dir}/{os.path.basename(input_video_name)}_{output_suffix}.mp4'
    output_video_writer = cv2.VideoWriter(
        output_video_path, fourcc, FPS,
        (output_frame_width, output_frame_height))
    if verbose:
        print(
            f'[INFO] INPUT: {input_video_path} ({frame_width}x{frame_height}@{FPS:.2f})'
        )
        print(f'[INFO] OUTPUT: {output_video_path}')

    return output_video_writer, (output_frame_width, output_frame_height)


def detect_face(image, detector, confidence_threshold=0.6):
    faces = detector.detect(image)
    # Filter face by confidence
    faces = [face for face in faces if face.confidence > confidence_threshold]
    return faces


def align_face(image, faces, image_size=112):
    aligned_faces = []
    for face in faces:
        landmarks = np.array(face.landmarks)
        aligned_face, _ = face_alignment(image,
                                         landmarks,
                                         image_size=image_size)
        aligned_faces.append(aligned_face)
    return aligned_faces


def align_face_and_embed(image, recognizer, faces, image_size=112):
    aligned_faces, embeddings = [], []
    for face in faces:
        landmarks = np.array(face.landmarks)
        aligned_face, _ = face_alignment(image,
                                         landmarks,
                                         image_size=image_size)
        aligned_faces.append(aligned_face)
        embedding = recognizer.get_normalized_embedding(image, landmarks)
        embeddings.append(embedding)
    return aligned_faces, embeddings


def match_face_by_similarity(embeddings,
                             target_embeddings,
                             similarity_threshold=0.3):
    match_id_dict = {i: None for i in range(len(embeddings))}
    max_match_similarities = [-1 for _ in range(len(embeddings))]
    for i, embedding in enumerate(embeddings):
        max_match_similarity = -1
        for j, target_embedding in enumerate(target_embeddings):
            similarity = compute_similarity(embedding, target_embedding)
            if similarity < similarity_threshold:
                continue
            if similarity > max_match_similarity:
                match_id_dict[i] = j
                max_match_similarity = similarity
                max_match_similarities[i] = max_match_similarity
    return match_id_dict, max_match_similarities


def get_matched_faces(faces, match_id_dict):
    matched_bboxes, matched_scores, matched_landmarks = [], [], []
    non_matched_bboxes, non_matched_scores, non_matched_landmarks = [], [], []
    for i, face in enumerate(faces):
        matched_target_id = match_id_dict[i]
        if matched_target_id is not None:
            matched_bboxes.append(face.bbox)
            matched_scores.append(face.confidence)
            matched_landmarks.append(face.landmarks)
        else:
            non_matched_bboxes.append(face.bbox)
            non_matched_scores.append(face.confidence)
            non_matched_landmarks.append(face.landmarks)
    return \
        (matched_bboxes, matched_scores, matched_landmarks), \
        (non_matched_bboxes, non_matched_scores, non_matched_landmarks)


def predict_age_gender(image, age_gender, faces):
    gender_strs, ages = [], []
    for face in faces:
        bbox = [int(ele) for ele in face.bbox]
        x1, y1, x2, y2 = bbox
        # Detect age/gender
        result = age_gender.predict(image, face.bbox)
        gender_str = 'F' if result.gender == 0 else 'M'
        gender_strs.append(gender_str)
        ages.append(result.age)
    return ages, gender_strs


def predict_emotion(image, emotion_predictor, faces):
    emotions = []
    for face in faces:
        bbox = [int(ele) for ele in face.bbox]
        x1, y1, x2, y2 = bbox
        # Predict emotion w/ landmarks
        landmarks = np.array(face.landmarks)
        result = emotion_predictor.predict(image, landmarks)
        emotions.append(result.emotion)
    return emotions


def estimate_gaze(image, gaze_estimator, faces):
    pitchs, yaws = [], []
    for face in faces:
        bbox = [int(ele) for ele in face.bbox]
        x1, y1, x2, y2 = bbox
        face_crop = image[y1:y2, x1:x2]
        if not 0 in face_crop.shape[:2]:
            result = gaze_estimator.estimate(face_crop)
            pitchs.append(result.pitch)
            yaws.append(result.yaw)
        else:
            pitchs.append(None)
            yaws.append(None)
    return pitchs, yaws


def parse_face(image, parser, aligned_faces):
    aligned_face_masks, aligned_face_masks_vis = [], []
    for aligned_face in aligned_faces:
        aligned_face_mask = parser.parse(aligned_face)
        aligned_face_mask_vis = vis_parsing_maps(aligned_face,
                                                 aligned_face_mask,
                                                 save_image=False)
        aligned_face_masks.append(aligned_face_mask)
        aligned_face_masks_vis.append(aligned_face_mask_vis)
    return aligned_face_masks, aligned_face_masks_vis
