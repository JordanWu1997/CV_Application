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

import time

import cv2
from tqdm import tqdm
from uniface import MobileGaze
from uniface.attribute.age_gender import AgeGender
from uniface.attribute.emotion import Emotion
from uniface.constants import (DDAMFNWeights, GazeWeights, ParsingWeights,
                               RetinaFaceWeights, YOLOv5FaceWeights)
from uniface.detection import RetinaFace, YOLOv5Face
from uniface.parsing import BiSeNet
from uniface.visualization import draw_detections, draw_gaze, vis_parsing_maps

from face_utils import (align_face, detect_face, estimate_gaze,
                        generate_output_video_writer, overlay_image_smart,
                        parse_face, predict_age_gender, predict_emotion)


def run_visualization_pipeline(canvas,
                               vis_threshold=0.6,
                               output_frame_width=-1,
                               output_frame_height=-1,
                               face_image_size=112,
                               add_history=True,
                               do_face_parsing=True):

    # Draw history
    overlay_x, overlay_y, gap = 0, frame_height - face_image_size, 0
    for i, history in enumerate(history_list):
        canvas = overlay_image_smart(canvas, history, int(overlay_x),
                                     int(overlay_y))
        cv2.putText(canvas, f'{frame_num:d}', (overlay_x + 2, overlay_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(canvas, f'{frame_num:d}', (overlay_x, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        overlay_x = overlay_x + face_image_size + gap

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
        if overlay_x + aligned_face_w > frame_width:
            overlay_x = x1 - padding - aligned_face_w
        canvas = overlay_image_smart(canvas, aligned_face_vis, int(overlay_x),
                                     int(overlay_y))
        # Update history
        if frame_num >= infer_frame_interval and frame_num % infer_frame_interval == 0:
            if len(history_list) >= \
                    frame_width // face_image_size + 0:
                history_list.pop(0)
            history_list.append(aligned_face_vis)

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
            if overlay_x + aligned_face_mask_w > frame_width:
                overlay_x = x1 - padding - aligned_face_mask_w
            canvas = overlay_image_smart(canvas, aligned_face_mask_vis,
                                         int(overlay_x), int(overlay_y))

    # Add OSD
    OSD_text = f'[INFO] Detected Faces: {len(bboxes):d} '
    OSD_text += f'FPS: {1/FD_elapse:.1f} '
    OSD_text += f'Infer every: {infer_frame_interval:d} frame '

    # Add OSD
    cv2.putText(canvas, OSD_text, (5, 30), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Resize
    if output_frame_width > 0 and output_frame_height > 0:
        canvas = cv2.resize(canvas, (output_frame_width, output_frame_height))

    return canvas


if __name__ == '__main__':

    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description="Process frames.")
    parser.add_argument('-i',
                        '--input',
                        nargs='+',
                        required=True,
                        help='Input video paths')
    parser.add_argument('-o',
                        '--output_dir',
                        default='./output',
                        help='Output video directory')
    parser.add_argument('-I',
                        '--infer_frame_interval',
                        default=5,
                        type=int,
                        help='Infer every infer_frame_interval frames')
    parser.add_argument('-p',
                        '--parse_face',
                        action='store_true',
                        help='Face component segmenation')
    args = parser.parse_args()

    # Input arguments
    input_video_paths = args.input
    output_video_dir = args.output_dir
    do_face_parsing = args.parse_face
    infer_frame_interval = max(args.infer_frame_interval, 1)
    output_suffix = 'output_FD'
    output_frame_width = -1
    live_display_frame_width = 720
    live_display = True
    debug = False

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
    for input_video_path in input_video_paths:
        cap = cv2.VideoCapture(input_video_path)

        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Generate output video writer
        output_video_writer, (output_frame_width, output_frame_height) = \
            generate_output_video_writer(input_video_path, cap,
                                         output_video_dir=output_video_dir,
                                         output_suffix=output_suffix,
                                         output_frame_width=output_frame_width,
                                         verbose=True)

        # Main
        progress_bar = tqdm(total=total_frames)
        frame_num = 0
        history_list = []
        while cap.isOpened():

            # Load frame
            ret, frame = cap.read()
            if not ret:
                break

            # Detect every specified frames
            if frame_num >= infer_frame_interval and \
                    frame_num % infer_frame_interval == 0:

                # Detect face
                FD_start = time.time()
                faces = detect_face(frame, detector, confidence_threshold=0.6)

                # Age, Gender
                age_gender_start = time.time()
                ages, gender_strs = predict_age_gender(frame, age_gender,
                                                       faces)

                # Emotion
                emotion_start = time.time()
                emotions = predict_emotion(frame, emotion_predictor, faces)

                # Gaze
                gaze_start = time.time()
                pitchs, yaws = estimate_gaze(frame, gaze_estimator, faces)

                # Align detected face (returns aligned image and inverse transform matrix)
                align_start = time.time()
                aligned_faces = align_face(frame, faces)

                # Parse aligned face
                parse_start = time.time()
                if do_face_parsing:
                    aligned_face_masks, aligned_face_masks_vis = \
                        parse_face(frame, parser, aligned_faces)

                FD_elapse = time.time() - FD_start

            # Visualization
            canvas = frame.copy()
            try:
                canvas = run_visualization_pipeline(
                    canvas,
                    output_frame_width=output_frame_width,
                    output_frame_height=output_frame_height,
                    do_face_parsing=do_face_parsing)
            except NameError as error:
                if debug:
                    print(error)

            # Save output
            output_video_writer.write(canvas)

            # Live display
            if live_display:
                canvas_height, canvas_width, _ = canvas.shape
                if live_display_frame_width > 0:
                    live_display_frame_height = \
                        int(live_display_frame_width * (canvas_height / canvas_width))
                    canvas = cv2.resize(
                        canvas,
                        (live_display_frame_width, live_display_frame_height))
                cv2.imshow(f'{input_video_path}', canvas)
                key = cv2.waitKey(1)
                if key == 27:  # Esc
                    break

            # Update progress
            progress_bar.update(1)
            frame_num += 1

        # Clear all cv2 windows
        if live_display:
            cv2.destroyAllWindows()
        cap.release()
        output_video_writer.release()
