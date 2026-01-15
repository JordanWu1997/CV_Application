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
from ultralytics import YOLO
from uniface import MobileGaze
from uniface.attribute.age_gender import AgeGender
from uniface.attribute.emotion import Emotion
from uniface.constants import (ArcFaceWeights, DDAMFNWeights, GazeWeights,
                               ParsingWeights, RetinaFaceWeights,
                               SphereFaceWeights, YOLOv5FaceWeights)
from uniface.detection import RetinaFace, YOLOv5Face
from uniface.parsing import BiSeNet
from uniface.recognition import ArcFace
from uniface.visualization import draw_detections, draw_gaze, vis_parsing_maps

from utils.face_utils import (align_face_and_embed, detect_face,
                              generate_output_video_writer, get_matched_faces,
                              get_target_faces_and_embeddings,
                              match_face_by_similarity, overlay_image_smart)


def run_visualization_pipeline(canvas,
                               output_frame_width=-1,
                               output_frame_height=-1,
                               vis_threshold=0.6,
                               face_image_size=112):

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
    overlay_x, overlay_y, gap = 0, frame_height - face_image_size, 6
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
        if overlay_x + aligned_face_w > frame_width:
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
        if overlay_x + aligned_face_w > frame_width:
            overlay_x = x1 - padding - aligned_face_w
        canvas = overlay_image_smart(canvas, target_aligned_face_vis,
                                     int(overlay_x), int(overlay_y))

        # Target result
        overlay_x = x2 + padding
        overlay_y = y1 + aligned_face_h + target_aligned_face_h + padding + gap + 20
        if overlay_x + aligned_face_w > frame_width:
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
    OSD_text += f'FPS: {1/FS_elapse:.1f} '
    OSD_text += f'Infer every: {infer_frame_interval:d} frame '
    OSD_text += f'Similarity THR: {similarity_threshold:.1f}'

    # Add OSD
    cv2.putText(canvas, OSD_text, (5, 30), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Resize
    if output_frame_width > 0 and output_frame_height > 0:
        canvas = cv2.resize(canvas, (output_frame_width, output_frame_height))

    return canvas


if __name__ == '__main__':

    import argparse

    class GroupArgsAction(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            # Get the current value of the attribute, or an empty list if it's the first time
            items = getattr(namespace, self.dest) or []
            # Append the new group of values as a sub-list
            items.append(values)
            # Set the attribute back to the namespace
            setattr(namespace, self.dest, items)

    # Input arguments
    parser = argparse.ArgumentParser(description="Process frames.")
    parser.add_argument('-i',
                        '--input',
                        nargs='+',
                        required=True,
                        help='Input video paths')

    parser.add_argument(
        "-t",
        "--target",
        dest="target_group",
        action=GroupArgsAction,
        nargs="+",
        help=
        "Target image group paths (use multiple -t if you have multiple targets)"
    )

    # Add an option for output directory (default value provided)
    parser.add_argument('-o',
                        '--output_dir',
                        default='./output',
                        help='Output video directory')
    parser.add_argument('-I',
                        '--infer_frame_interval',
                        default=5,
                        type=int,
                        help='Infer every infer_frame_interval frames')
    parser.add_argument('-s',
                        '--similarity_threshold',
                        default=0.3,
                        type=float,
                        help='Similarity threshold')
    args = parser.parse_args()

    print(args.target)
    import sys
    sys.exit()

    # Parse input arguments
    input_video_paths = args.input
    target_image_paths_group = args.target_group
    output_video_dir = args.output_dir
    infer_frame_interval = max(args.infer_frame_interval, 1)
    similarity_threshold = args.similarity_threshold
    output_suffix = 'output_FS'
    output_frame_width = -1
    live_display_frame_width = 720
    live_display = True
    debug = False

    # Init pose esimator
    model_weight = './weights/yolo11x-pose.pt'
    pose_model = YOLO(model_weight)

    # Init model: detector
    detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34,
                          conf_thresh=0.5,
                          nms_thresh=0.4,
                          input_size=(640, 640))
    # # Init model: detector
    # detector = YOLOv5Face(model_name=YOLOv5FaceWeights.YOLOV5S,
    # conf_thresh=0.6,
    # nms_thresh=0.5)

    # Init model: recognizer
    recognizer = ArcFace(model_name=ArcFaceWeights.RESNET)

    # Target images
    target_aligned_faces_group, target_embeddings_groups = []
    for target_image_paths in target_image_paths_group:
        target_aligned_faces, target_embeddings = \
            get_target_faces_and_embeddings(target_image_paths, detector, recognizer)

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
        progress_bar, frame_num = tqdm(total=total_frames), 0
        while cap.isOpened():

            # Load frame
            ret, frame = cap.read()
            if not ret:
                break

            # Inference pipeline
            if frame_num >= infer_frame_interval and \
                    frame_num % infer_frame_interval == 0:

                # Post estimate
                pose_start = time.time()
                pose_results = pose_model.track(frame,
                                                persist=True,
                                                verbose=False)

                # Detect face
                FS_start = time.time()
                faces = detect_face(frame, detector, confidence_threshold=0.6)

                # Embed face
                align_start = time.time()
                aligned_faces, embeddings = \
                    align_face_and_embed(frame, recognizer, faces)

                # Match by similarity and get matched faces
                match_start = time.time()
                match_id_dict, max_match_similarities = \
                    match_face_by_similarity(embeddings, target_embeddings, similarity_threshold=0.3)
                (matched_bboxes, matched_scores, matched_landmarks), \
                (non_matched_bboxes, non_matched_scores, non_matched_landmarks) = \
                    get_matched_faces(faces, match_id_dict)

                FS_elapse = time.time() - FS_start

            # Visualization
            canvas = frame.copy()
            try:
                canvas = run_visualization_pipeline(
                    canvas,
                    output_frame_width=output_frame_width,
                    output_frame_height=output_frame_height)
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
