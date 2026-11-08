#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Search video frames for faces represented by one or more target groups.

Example:
    python do_video_FS.py -i clip.mp4 --target-group staff=./targets/staff \
        --target-group visitors=./targets/visitors

Each target directory can contain images directly or in nested directories.
"""

import argparse
import math
import os
import time

import cv2
import numpy as np
from tqdm import tqdm
from uniface import MobileGaze, compute_similarity
from uniface.constants import (ArcFaceWeights, GazeWeights, HeadPoseWeights,
                               RetinaFaceWeights)
from uniface.detection import RetinaFace
from uniface.face_utils import face_alignment
from uniface.headpose import HeadPose
from uniface.recognition import ArcFace

from utils.face_pipeline import (FaceFrame, FaceRecord, FaceTracker,
                                 TrackIdentityVoter, crop_face, output_layout,
                                 render_face_frame)
from utils.face_utils import detect_face, estimate_gaze, estimate_head_pose
from utils.inference_device import (model_provider_kwargs, resolve_providers,
                                    uniface_providers, verify_model_providers)
from utils.video_writer import EncodedVideoWriter, add_encoding_arguments

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _target_images(path):
    if os.path.isfile(path):
        return [path] if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS else []
    if not os.path.isdir(path):
        return []
    files = []
    for root, dirs, names in os.walk(path):
        dirs.sort()
        files.extend(os.path.join(root, name) for name in sorted(names)
                     if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS)
    return files


def _load_target_group(name, path, detector, recognizer, confidence_threshold):
    group = {'name': name, 'faces': [], 'embeddings': [], 'labels': []}
    paths = _target_images(path)
    if not paths:
        raise ValueError(f"Target group '{name}' has no readable image files: {path}")
    for image_path in paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f'[WARN] Could not read target image: {image_path}')
            continue
        faces = detect_face(image, detector,
                            confidence_threshold=confidence_threshold)
        for face_idx, face in enumerate(faces, start=1):
            landmarks = np.asarray(face.landmarks)
            aligned, _ = face_alignment(image, landmarks, image_size=112)
            embedding = recognizer.get_normalized_embedding(image, landmarks)
            group['faces'].append(aligned)
            group['embeddings'].append(embedding)
            suffix = f' #{face_idx}' if len(faces) > 1 else ''
            group['labels'].append(os.path.basename(image_path) + suffix)
    if not group['embeddings']:
        raise ValueError(f"Target group '{name}' contains no detected faces: {path}")
    print(f"[INFO] Target group '{name}': {len(group['embeddings'])} face(s) from {len(paths)} image(s)")
    return group


def _match_faces(embeddings, target_groups, threshold):
    matches, similarities = [], []
    for embedding in embeddings:
        best = (None, None)
        best_similarity = -1.0
        for group_idx, group in enumerate(target_groups):
            for target_idx, target_embedding in enumerate(group['embeddings']):
                similarity = float(compute_similarity(embedding, target_embedding))
                if similarity > best_similarity:
                    best, best_similarity = (group_idx, target_idx), similarity
        matches.append(best if best_similarity >= threshold else (None, None))
        similarities.append(best_similarity)
    return matches, similarities


class FaceSearchPipeline:
    """Reusable FS processor. Call process_frame before drawing or downstream fusion."""

    def __init__(self, detector, recognizer, target_groups, threshold=0.3,
                 confidence_threshold=0.6, infer_interval=5, tracking=False,
                 gaze_estimator=None, vote_window=5, min_votes=3,
                 head_pose_estimator=None):
        self.detector = detector
        self.recognizer = recognizer
        self.target_groups = target_groups
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        if infer_interval < 1:
            raise ValueError('infer_interval must be at least 1.')
        self.infer_interval = infer_interval
        self.tracker = FaceTracker() if tracking else None
        self.voter = TrackIdentityVoter(vote_window, min_votes) if tracking else None
        self.gaze_estimator = gaze_estimator
        self.head_pose_estimator = head_pose_estimator
        self.last = FaceFrame(-1)

    def process_frame(self, frame, frame_index):
        inferred = frame_index % self.infer_interval == 0 or self.last.frame_index < 0
        if inferred:
            start = time.perf_counter()
            faces = detect_face(frame, self.detector,
                                confidence_threshold=self.confidence_threshold)
            embeddings = [self.recognizer.get_normalized_embedding(
                frame, np.asarray(face.landmarks)) for face in faces]
            matches, similarities = _match_faces(embeddings, self.target_groups,
                                                  self.threshold)
            gaze = (estimate_gaze(frame, self.gaze_estimator, faces)
                    if self.gaze_estimator else ([None] * len(faces), [None] * len(faces)))
            records = []
            for index, (face, match, similarity) in enumerate(zip(faces, matches, similarities)):
                group_idx, target_idx = match
                group = self.target_groups[group_idx] if group_idx is not None else None
                pose = (estimate_head_pose(frame, self.head_pose_estimator, face)
                        if self.head_pose_estimator is not None else None)
                attributes = {
                    'matched': group is not None,
                    'group': group['name'] if group else 'NO MATCH',
                    'reference_index': target_idx if group else None,
                    'reference_label': group['labels'][target_idx] if group else '',
                    'similarity': similarity,
                    'pitch': gaze[0][index], 'yaw': gaze[1][index],
                }
                if pose is not None:
                    attributes.update(head_pitch=pose.pitch, head_yaw=pose.yaw,
                                      head_roll=pose.roll)
                records.append(FaceRecord(
                    bbox=tuple(face.bbox), landmarks=np.asarray(face.landmarks),
                    confidence=float(face.confidence), crop=crop_face(frame, face.bbox),
                    reference_crop=group['faces'][target_idx] if group else None,
                    attributes=attributes, extras={'group_index': group_idx}))
            if self.tracker:
                records = self.tracker.update(frame, records)
                records = self.voter.update(records, self.target_groups)
            result = FaceFrame(frame_index, records, True,
                               time.perf_counter() - start, self.tracker is not None)
        else:
            records = (self.tracker.update(frame) if self.tracker else self.last.faces)
            result = FaceFrame(frame_index, records, False,
                               self.last.inference_seconds, self.tracker is not None)
        self.last = result
        return result


def _parse_args():
    parser = argparse.ArgumentParser(description='Search videos for target face groups.')
    parser.add_argument('-i', '--input', nargs='+', required=True, help='Input video paths')
    parser.add_argument('--target-group', action='append', default=[], metavar='NAME=PATH',
                        help='Named target directory or image; repeat for multiple groups')
    parser.add_argument('-t', '--target', action='append', nargs='+', default=[],
                        help='Legacy target image paths; repeat -t to create groups')
    parser.add_argument('-o', '--output_dir', default='./output', help='Output video directory')
    parser.add_argument('-I', '--infer_frame_interval', default=5, type=int,
                        help='Run face detection every N frames')
    parser.add_argument('-s', '--similarity_threshold', default=0.3, type=float,
                        help='Cosine similarity threshold for a match')
    parser.add_argument('--confidence-threshold', default=0.6, type=float,
                        help='Minimum face detector confidence')
    parser.add_argument('--detector-input-size', default=640, type=int,
                        help='RetinaFace square input size')
    parser.add_argument('--no-display', action='store_true', help='Disable live preview')
    parser.add_argument('--display-width', type=int, default=0,
                        help='Live preview width; 0 keeps full resolution')
    parser.add_argument('--track-faces', action='store_true',
                        help='Track faces and vote on FS matches across detections')
    parser.add_argument('--track-vote-window', type=int, default=5,
                        help='Number of detections per track used for match voting')
    parser.add_argument('--track-min-votes', type=int, default=3,
                        help='Matching detections required before showing an identity')
    parser.add_argument('--gaze', action='store_true',
                        help='Estimate gaze and draw its arrow (default: off)')
    parser.add_argument('--no-gaze', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--show-landmarks', action='store_true',
                        help='Draw five detector landmarks (default: off)')
    parser.add_argument('--head-pose', action='store_true',
                        help='Estimate head pose and draw its 3D box')
    parser.add_argument('--result-layout', choices=('panel', 'box'), default='panel',
                        help='Attached panel or frame-only cards beside faces')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'openvino'),
                        default='auto',
                        help='ONNX device; auto prefers CoreML, CUDA, OpenVINO, then CPU')
    parser.add_argument('--openvino-device', choices=('CPU', 'GPU'), default='CPU',
                        help='OpenVINO target device (default: CPU)')
    parser.add_argument('--openvino-precision', choices=('FP32', 'FP16'), default='FP32',
                        help='OpenVINO GPU inference precision hint (default: FP32)')
    add_encoding_arguments(parser)
    args = parser.parse_args()

    groups = []
    for spec in args.target_group:
        if '=' not in spec:
            parser.error(f"--target-group expects NAME=PATH, received: {spec}")
        name, path = spec.split('=', 1)
        if not name or not path:
            parser.error(f"--target-group expects NAME=PATH, received: {spec}")
        groups.append((name, [path]))
    for index, paths in enumerate(args.target, start=1):
        if paths:
            name = (os.path.basename(os.path.normpath(paths[0]))
                    if len(paths) == 1 else f'legacy group {index}')
            groups.append((name or f'group {index}', paths))
    if not groups:
        parser.error('Provide at least one --target-group NAME=DIR (or legacy -t images).')
    args.target_groups = groups
    return args


def main():
    args = _parse_args()
    if args.track_vote_window < 1 or not 1 <= args.track_min_votes <= args.track_vote_window:
        raise ValueError('Require 1 <= --track-min-votes <= --track-vote-window.')
    if args.infer_frame_interval < 1:
        raise SystemExit('--infer_frame_interval must be at least 1.')
    interval = args.infer_frame_interval
    confidence_threshold = min(max(args.confidence_threshold, 0.0), 1.0)
    providers = resolve_providers(args.device, args.openvino_device,
                                  args.openvino_precision)
    with uniface_providers(providers):
        detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34,
                              confidence_threshold=0.5, nms_threshold=0.4,
                              input_size=(args.detector_input_size,
                                          args.detector_input_size),
                              **model_provider_kwargs(RetinaFace, providers))
        verify_model_providers({'detector': detector}, providers)
        recognizer = ArcFace(model_name=ArcFaceWeights.RESNET,
                             **model_provider_kwargs(ArcFace, providers))
        gaze = (MobileGaze(model_name=GazeWeights.RESNET18,
                           **model_provider_kwargs(MobileGaze, providers))
                if args.gaze and not args.no_gaze else None)
        head_pose = (HeadPose(model_name=HeadPoseWeights.MOBILENET_V3_SMALL,
                              **model_provider_kwargs(HeadPose, providers))
                     if args.head_pose else None)
    models = {'detector': detector, 'recognizer': recognizer}
    if gaze is not None:
        models['gaze'] = gaze
    if head_pose is not None:
        models['head_pose'] = head_pose
    verify_model_providers(models, providers)
    openvino_setting = (f'; OpenVINO target: {args.openvino_device}; '
                        f'precision hint: {args.openvino_precision}'
                        if args.device == 'openvino' else '')
    print(f'[INFO] Inference device: {args.device}{openvino_setting}; ONNX providers: '
          f'{detector.session.get_providers()}')
    target_groups = []
    for name, paths in args.target_groups:
        combined = {'name': name, 'faces': [], 'embeddings': [], 'labels': []}
        for path in paths:
            group = _load_target_group(name, path, detector, recognizer,
                                       confidence_threshold)
            combined['faces'].extend(group['faces'])
            combined['embeddings'].extend(group['embeddings'])
            combined['labels'].extend(group['labels'])
        target_groups.append(combined)

    for input_path in args.input:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f'[WARN] Could not open video: {input_path}')
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(args.output_dir, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        layout = output_layout((height, width, 3), target_groups,
                               min_side_width=480,
                               result_layout=args.result_layout)
        pipeline = FaceSearchPipeline(detector, recognizer, target_groups,
                                      args.similarity_threshold, confidence_threshold,
                                      interval, args.track_faces, gaze,
                                      args.track_vote_window, args.track_min_votes,
                                      head_pose)
        output_path = os.path.join(args.output_dir,
                                   os.path.splitext(os.path.basename(input_path))[0] + '_output_FS.mp4')
        try:
            writer = EncodedVideoWriter(output_path, layout[:2], fps,
                                        args.video_codec, args.video_crf,
                                        args.video_preset)
        except Exception:
            cap.release()
            raise
        print(f'[INFO] OUTPUT: {output_path} ({layout[0]}x{layout[1]}@{fps:.2f}, '
              f'{args.video_codec} CRF {args.video_crf}, {args.video_preset})')
        frame_num = 0
        progress = tqdm(total=total_frames)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = pipeline.process_frame(frame, frame_num)
                canvas = render_face_frame(frame, result, 'FS', layout,
                                           target_groups, interval,
                                           result_layout=args.result_layout,
                                           show_gaze=gaze is not None,
                                           show_landmarks=args.show_landmarks,
                                           head_pose_enabled=args.head_pose)
                writer.write(canvas)
                if not args.no_display:
                    preview = canvas
                    if args.display_width > 0 and canvas.shape[1] > args.display_width:
                        height = int(args.display_width * canvas.shape[0] / canvas.shape[1])
                        preview = cv2.resize(canvas, (args.display_width, height))
                    cv2.imshow(os.path.basename(input_path), preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        return
                    if key == 27:
                        break
                frame_num += 1
                progress.update(1)
        finally:
            progress.close()
            cap.release()
            writer.release()
            if not args.no_display:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
