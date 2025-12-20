#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import sys
import time

import cv2
import mediapipe as mp

from utils import get_available_devices, parse_video_device, resize_image


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
    parser.add_argument('-s',
                        '--skip_frame',
                        type=int,
                        default=3,
                        help='Number of frame to skip')
    args = parser.parse_args()

    # List available devices
    if args.list_devices:
        devices = get_available_devices(verbose=True)
        sys.exit(f'[INFO] Found devices: {devices}')

    # 1. Initialize MediaPipe Holistic and Drawing Utilities
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 2. Setup Webcam
    input_device = parse_video_device(args.input_device, YT_URL=args.YT_URL)
    cap = cv2.VideoCapture(input_device)
    input_FPS = cap.get(cv2.CAP_PROP_FPS)

    # 3. Initialize Holistic Model
    counter, skip_frame, playspeed = 0, args.skip_frame, 2
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=
            True  # Enables iris tracking and detailed eye mesh
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Ignoring empty camera frame.")
                continue
            start = time.time()
            counter += 1

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
                    print(
                        '[ERROR] Cannot get image from source ... Retrying ...'
                    )
                    time.sleep(0.05)
                start = time.time()

            # Get image geometry: size
            height, width, _ = frame.shape

            # Resize frame
            if round(args.resize_ratio, 3) != 1.0:
                frame = resize_image(frame,
                                     width=width,
                                     height=height,
                                     resize_ratio=args.resize_ratio)

            if not counter % skip_frame == 0 and counter > skip_frame:
                # Display the output
                cv2.imshow('MediaPipe Holistic Tracking', frame)
                continue

            # Flip the image horizontally for a selfie-view display
            # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Performance optimization: mark image as not writeable
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            # Draw detections back on the original BGR frame
            # 1. Face Mesh
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.
                get_default_face_mesh_contours_style())

            # 2. Pose Detection
            mp_drawing.draw_landmarks(frame,
                                      results.pose_landmarks,
                                      mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.
                                      get_default_pose_landmarks_style())

            # 3. Hand Detection (Left and Right)
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Display the output
            cv2.imshow('MediaPipe Holistic Tracking', frame)

            # Exit on 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
