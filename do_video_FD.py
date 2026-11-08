#!/usr/bin/env python3
"""Video face detection and attributes with optional tracking and separate rendering."""

import argparse
from inspect import signature
import math
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm
from uniface import MobileGaze
from uniface.attribute.age_gender import AgeGender
from uniface.attribute.emotion import Emotion
from uniface.attribute import FaceAttribNet
from uniface.constants import (EDifFIQAWeights, GazeWeights, ParsingWeights,
                               RetinaFaceWeights)
try:
    from uniface.constants import EmotionWeights
except ImportError:  # UniFace 2.0
    from uniface.constants import DDAMFNWeights as EmotionWeights
from uniface.detection import RetinaFace
from uniface.parsing import BiSeNet
from uniface.quality import EDifFIQA

from utils.face_pipeline import (FaceFrame, FaceRecord, FaceTracker,
                                 TrackAttributeVoter,
                                 output_layout, render_face_frame)
from utils.face_utils import (align_face, detect_face, estimate_gaze,
                              parse_face, predict_age_gender, predict_emotion)
from utils.inference_device import (model_provider_kwargs, resolve_providers,
                                    uniface_providers, verify_model_providers)
from utils.video_writer import EncodedVideoWriter, add_encoding_arguments


class DeviceEmotion(Emotion):
    """Place UniFace's TorchScript emotion model on the requested torch device."""

    def __init__(self, *args, device, **kwargs):
        self.requested_device = device
        super().__init__(*args, **kwargs)

    def _initialize_model(self):
        self.device = torch.device(self.requested_device)
        super()._initialize_model()


class FaceDetectionPipeline:
    """Reusable FD processor. process_frame returns data independent of rendering."""

    def __init__(self, detector, age_gender, emotion_predictor, gaze_estimator,
                 parser=None, confidence_threshold=0.6, infer_interval=5,
                 tracking=False, face_states=None, quality_estimator=None,
                 vote_window=5):
        self.detector = detector
        self.age_gender = age_gender
        self.emotion_predictor = emotion_predictor
        self.gaze_estimator = gaze_estimator
        self.parser = parser
        self.face_states = face_states
        self.quality_estimator = quality_estimator
        self.confidence_threshold = confidence_threshold
        self.infer_interval = max(1, infer_interval)
        self.tracker = FaceTracker() if tracking else None
        self.voter = TrackAttributeVoter(vote_window) if tracking else None
        self.last = FaceFrame(-1)

    def process_frame(self, frame, frame_index):
        inferred = frame_index % self.infer_interval == 0 or self.last.frame_index < 0
        if inferred:
            start = time.perf_counter()
            faces = detect_face(frame, self.detector,
                                confidence_threshold=self.confidence_threshold)
            ages, genders = predict_age_gender(frame, self.age_gender, faces)
            emotions = (predict_emotion(frame, self.emotion_predictor, faces)
                        if self.emotion_predictor else ['N/A'] * len(faces))
            pitches, yaws = estimate_gaze(frame, self.gaze_estimator, faces)
            aligned = align_face(frame, faces)
            masks = parse_face(frame, self.parser, aligned)[1] if self.parser else [None] * len(faces)
            records = []
            for index, face in enumerate(faces):
                attributes = {
                    'age': ages[index], 'gender': genders[index],
                    'emotion': emotions[index], 'pitch': pitches[index],
                    'yaw': yaws[index],
                }
                if self.face_states is not None:
                    states = self.face_states.predict(frame, face)
                    attributes['face_states'] = states.as_dict()
                if self.quality_estimator is not None:
                    attributes['quality_score'] = self.quality_estimator.score_aligned(
                        aligned[index]).score
                records.append(FaceRecord(
                    bbox=tuple(face.bbox), landmarks=np.asarray(face.landmarks),
                    confidence=float(face.confidence), crop=aligned[index],
                    mask=masks[index], attributes=attributes))
            if self.tracker:
                records = self.tracker.update(frame, records)
                records = self.voter.update(records)
            result = FaceFrame(frame_index, records, True,
                               time.perf_counter() - start, self.tracker is not None)
        else:
            records = self.tracker.update(frame) if self.tracker else self.last.faces
            result = FaceFrame(frame_index, records, False,
                               self.last.inference_seconds, self.tracker is not None)
        self.last = result
        return result


def _parse_args():
    parser = argparse.ArgumentParser(description='Detect faces and attributes in videos.')
    parser.add_argument('-i', '--input', nargs='+', required=True, help='Input video paths')
    parser.add_argument('-o', '--output_dir', default='./output', help='Output video directory')
    parser.add_argument('-I', '--infer_frame_interval', type=int, default=5,
                        help='Run detection every N frames')
    parser.add_argument('--confidence-threshold', type=float, default=0.6)
    parser.add_argument('--detector-input-size', type=int, default=640)
    parser.add_argument('-p', '--parse_face', action='store_true',
                        help='Add face parsing images to the detail panel')
    parser.add_argument('--no-emotion', action='store_true',
                        help='Skip the separate PyTorch emotion model')
    parser.add_argument('--face-states', action='store_true',
                        help='Predict eye, glasses, sunglasses, and mask states')
    parser.add_argument('--face-quality', action='store_true',
                        help='Score aligned face image quality')
    parser.add_argument('--track-faces', action='store_true',
                        help='Track faces and vote on FD attributes across detections')
    parser.add_argument('--track-vote-window', type=int, default=5,
                        help='Number of detections per track used for attribute voting')
    parser.add_argument('--no-display', action='store_true', help='Disable live preview')
    parser.add_argument('--display-width', type=int, default=0,
                        help='Live preview width; 0 keeps full resolution')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'openvino'),
                        default='auto',
                        help='ONNX device; auto prefers CoreML, CUDA, OpenVINO, then CPU')
    parser.add_argument('--openvino-device', choices=('CPU', 'GPU'), default='CPU',
                        help='OpenVINO target device (default: CPU)')
    parser.add_argument('--openvino-precision', choices=('FP32', 'FP16'), default='FP32',
                        help='OpenVINO GPU inference precision hint (default: FP32)')
    add_encoding_arguments(parser)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.track_vote_window < 1:
        raise ValueError('--track-vote-window must be at least 1.')
    interval = max(1, args.infer_frame_interval)
    threshold = min(max(args.confidence_threshold, 0.0), 1.0)
    providers = resolve_providers(args.device, args.openvino_device,
                                  args.openvino_precision)
    if args.device == 'cuda' and not args.no_emotion and not torch.cuda.is_available():
        raise RuntimeError('CUDA was selected, but PyTorch CUDA is unavailable for emotion inference.')
    with uniface_providers(providers):
        detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34,
                              confidence_threshold=0.5, nms_threshold=0.4,
                              input_size=(args.detector_input_size, args.detector_input_size),
                              **model_provider_kwargs(RetinaFace, providers))
        verify_model_providers({'detector': detector}, providers)
        age_gender = AgeGender(**model_provider_kwargs(AgeGender, providers))
        gaze = MobileGaze(model_name=GazeWeights.RESNET18,
                          **model_provider_kwargs(MobileGaze, providers))
        parser_kwargs = {'input_size': (112, 112)} if 'input_size' in signature(BiSeNet).parameters else {}
        face_parser = (BiSeNet(model_name=ParsingWeights.RESNET34,
                              **parser_kwargs, **model_provider_kwargs(BiSeNet, providers))
                       if args.parse_face else None)
        face_states = (FaceAttribNet(**model_provider_kwargs(FaceAttribNet, providers))
                       if args.face_states else None)
        quality_estimator = (EDifFIQA(model_name=EDifFIQAWeights.T,
                                     **model_provider_kwargs(EDifFIQA, providers))
                             if args.face_quality else None)
    models = {'detector': detector, 'age_gender': age_gender, 'gaze': gaze}
    if face_parser is not None:
        models['parsing'] = face_parser
    if face_states is not None:
        models['face_states'] = face_states
    if quality_estimator is not None:
        models['face_quality'] = quality_estimator
    verify_model_providers(models, providers)
    emotion = None
    if not args.no_emotion:
        torch_device = 'cpu' if args.device in ('cpu', 'openvino') else 'cuda'
        emotion_key = ('model_name' if 'model_name' in signature(Emotion).parameters
                       else 'model_weights')
        emotion_kwargs = {emotion_key: EmotionWeights.AFFECNET8,
                          'input_size': (112, 112)}
        if args.device == 'auto':
            emotion = Emotion(**emotion_kwargs)
        else:
            emotion = DeviceEmotion(**emotion_kwargs, device=torch_device)
    openvino_setting = (f'; OpenVINO target: {args.openvino_device}; '
                        f'precision hint: {args.openvino_precision}'
                        if args.device == 'openvino' else '')
    print(f'[INFO] Inference device: {args.device}{openvino_setting}; ONNX providers: '
          f'{detector.session.get_providers()}; emotion: '
          f'{emotion.device if emotion is not None else "disabled"}')

    for input_path in args.input:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f'[WARN] Could not open video: {input_path}')
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        layout = output_layout((height, width, 3))
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir,
                                   os.path.splitext(os.path.basename(input_path))[0] + '_output_FD.mp4')
        try:
            writer = EncodedVideoWriter(output_path, layout[:2], fps,
                                        args.video_codec, args.video_crf,
                                        args.video_preset)
        except Exception:
            cap.release()
            raise
        print(f'[INFO] OUTPUT: {output_path} ({layout[0]}x{layout[1]}@{fps:.2f}, '
              f'{args.video_codec} CRF {args.video_crf}, {args.video_preset})')
        pipeline = FaceDetectionPipeline(
            detector, age_gender, emotion, gaze,
            parser=face_parser, confidence_threshold=threshold,
            infer_interval=interval, tracking=args.track_faces,
            face_states=face_states, quality_estimator=quality_estimator,
            vote_window=args.track_vote_window)
        progress = tqdm(total=total)
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = pipeline.process_frame(frame, frame_index)
                canvas = render_face_frame(frame, result, 'FD', layout,
                                           infer_interval=interval)
                writer.write(canvas)
                if not args.no_display:
                    preview = canvas
                    if args.display_width > 0 and canvas.shape[1] > args.display_width:
                        preview_height = int(args.display_width * canvas.shape[0] / canvas.shape[1])
                        preview = cv2.resize(canvas, (args.display_width, preview_height))
                    cv2.imshow(os.path.basename(input_path), preview)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                frame_index += 1
                progress.update(1)
        finally:
            progress.close()
            cap.release()
            writer.release()
            if not args.no_display:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
