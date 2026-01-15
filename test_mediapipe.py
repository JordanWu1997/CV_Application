#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import sys
import time

import cv2
import mediapipe as mp
from ultralytics import YOLO

from utils.utils import (get_available_devices, parse_video_device,
                         put_text_to_canvas, resize_image, toggle_bool_option)


class PersonSession:

    def __init__(self):
        # Each person gets their own dedicated MediaPipe instances
        self.mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True)
        self.last_seen = time.time()

    def process(self, crop):
        self.last_seen = time.time()
        # MediaPipe logic here
        return self.mp_holistic.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def main():
    """  """

    import argparse

    # Input argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--list-devices',
                        action='store_true',
                        help='Get available device list')
    parser.add_argument('-i',
                        '--input_device',
                        default=None,
                        type=str,
                        help='Input device, file or strearming URL')
    parser.add_argument('-y',
                        '--YT_URL',
                        help='If input URL is youtube URL',
                        action='store_true')
    parser.add_argument('-r',
                        '--resize_ratio',
                        default=1.0,
                        type=float,
                        help='Ratio to resize live display')
    parser.add_argument('-f',
                        '--start_frame',
                        default=0,
                        type=int,
                        help='Frame to start')
    parser.add_argument('-I',
                        '--infer_frame_interval',
                        type=int,
                        default=3,
                        help='Number of frame to skip')
    args = parser.parse_args()

    # List available devices
    if args.list_devices:
        devices = get_available_devices(verbose=True)
        sys.exit(f'[INFO] Found devices: {devices}')

    # COCO object detector as human detector
    human_detector = YOLO('./weights/yolo11x.pt')

    # 1. Initialize MediaPipe Holistic and Drawing Utilities
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 2. Setup Webcam
    input_device = parse_video_device(args.input_device, YT_URL=args.YT_URL)
    cap = cv2.VideoCapture(input_device)
    input_FPS = cap.get(cv2.CAP_PROP_FPS)

    active_trackers = {}
    cleanup_threshold = 15
    frame_num, infer_frame_interval, playspeed, show_OSD = 0, args.infer_frame_interval, 1, True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # print("[INFO] Ignoring empty camera frame.")
            # continue
            break
        start = time.time()
        frame_num += 1

        # Speedup playing
        if playspeed > 1:
            for _ in range(int(playspeed) - 1):
                ret, frame = cap.read()
                if not ret:
                    print(
                        '[ERROR] Cannot get image from source ... Retrying ...'
                    )
                    time.sleep(0.1)
            start = time.time()
        elif playspeed < 1 and playspeed > 0:
            FPS = input_FPS * playspeed
            time.sleep(1 / FPS)
            ret, frame = cap.read()
            if not ret:
                print('[ERROR] Cannot get image from source ... Retrying ...')
                time.sleep(0.05)
            start = time.time()

        # Flip for camera input
        if isinstance(input_device, int):
            frame = cv2.flip(frame, 1)

        # Get image geometry: size
        frame_height, frame_width, _ = frame.shape

        # Resize frame
        if round(args.resize_ratio, 3) != 1.0:
            frame = resize_image(frame,
                                 width=frame_width,
                                 height=frame_height,
                                 resize_ratio=args.resize_ratio)

        # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inference
        if frame_num % infer_frame_interval == 0 \
                and frame_num > infer_frame_interval:

            # Detect human
            human_results = human_detector.track(frame,
                                                 persist=True,
                                                 classes=[0],
                                                 conf=0.6,
                                                 verbose=False,
                                                 tracker='botsort.yaml')

            # Mediapipe
            mediapipe_results, person_bboxes, person_image_rgbs = [], [], []
            for human_result in human_results:
                if not (human_result.boxes is not None
                        and human_result.boxes.id is not None):
                    continue
                bboxes = human_result.boxes.xyxy.cpu().numpy()
                track_ids = human_result.boxes.id.int().cpu().numpy()
                for bbox, track_id in zip(bboxes, track_ids):
                    # Get person bbox
                    x1 = max(int(bbox[0]), 0)
                    y1 = max(int(bbox[1]), 0)
                    x2 = min(int(bbox[2]), frame_width - 1)
                    y2 = min(int(bbox[3]), frame_height - 1)

                    # Initialize Holistic Model: Create session if it's a new ID
                    if track_id not in active_trackers:
                        active_trackers[track_id] = PersonSession()

                    # Crop person region
                    person_image_rgb = image_rgb[y1:y2, x1:x2]
                    person_image_rgbs.append(person_image_rgb)
                    person_bboxes.append([x1, y1, x2, y2])

                    # Performance optimization: mark image as not writeable
                    image_rgb.flags.writeable = False
                    results = \
                        active_trackers[track_id].process(person_image_rgb)
                    image_rgb.flags.writeable = True

                    # Collect mediapipe result
                    mediapipe_results.append(results)

            # Infer FPS
            infer_FPS = 1 / (time.time() - start)

        # Clean up old IDs
        expired_ids = [
            tid for tid, obj in active_trackers.items()
            if time.time() - obj.last_seen > cleanup_threshold
        ]
        for tid in expired_ids:
            active_trackers[tid].mp_holistic.close()  # Free memory
            del active_trackers[tid]

        # Visualization: init canvas
        canvas = frame.copy()
        try:

            # Mediapipe visualization
            for results, person_bbox in zip(mediapipe_results, person_bboxes):
                x1, y1, x2, y2 = person_bbox
                person_image_rgb = image_rgb[y1:y2, x1:x2]

                # Draw detections back on the original BGR frame
                # 1. Face Mesh
                mp_drawing.draw_landmarks(
                    person_image_rgb,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.
                    get_default_face_mesh_contours_style())

                # 2. Pose Detection
                mp_drawing.draw_landmarks(
                    person_image_rgb,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.
                    get_default_pose_landmarks_style())

                # 3. Hand Detection (Left and Right)
                mp_drawing.draw_landmarks(
                    person_image_rgb, results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                mp_drawing.draw_landmarks(
                    person_image_rgb, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Update person image
                person_image_bgr = cv2.cvtColor(person_image_rgb,
                                                cv2.COLOR_RGB2BGR)
                canvas[y1:y2, x1:x2] = person_image_bgr

            # Human detection visualization
            canvas = human_result.plot(img=canvas)

        except NameError:
            pass

        try:
            infer_FPS
        except NameError:
            infer_FPS = -1

        # Add OSD
        OSD_text = f'Input FPS: {input_FPS:.1f}, '
        OSD_text += f'Infer FPS: {infer_FPS:.1f}, '
        OSD_text += f'Playspeed: {playspeed:.2f}, '
        OSD_text += f'Infer every {infer_frame_interval:d} frame'
        if show_OSD:
            put_text_to_canvas(canvas,
                               OSD_text,
                               top_left=(10, 30),
                               font_scale=0.5,
                               fg_color=(0, 255, 0),
                               thickness=1)

        # Display the output
        cv2.imshow(
            'Ultralytics YOLOv11 Person Detection + MediaPipe Holistic Tracking',
            canvas)

        # Refresh
        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
            break
        if key == 13:  # Enter
            show_OSD = toggle_bool_option(show_OSD)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
