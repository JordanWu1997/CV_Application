#!/usr/bin/env python3
"""Video face detection and attributes with optional tracking and separate rendering."""

import argparse
import math
import os
import time
from inspect import signature

import cv2
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm
from uniface import MobileGaze
from uniface.attribute import FaceAttribNet
from uniface.attribute.age_gender import AgeGender
from uniface.attribute.emotion import Emotion
from uniface.constants import (EDifFIQAWeights, GazeWeights, HeadPoseWeights,
                               ParsingWeights, RetinaFaceWeights)

try:
    from uniface.constants import EmotionWeights
except ImportError:  # UniFace 2.0
    from uniface.constants import DDAMFNWeights as EmotionWeights

from uniface.detection import RetinaFace

try:
    from uniface.draw import vis_parsing_maps
except ImportError:  # UniFace 2.0
    from uniface.visualization import vis_parsing_maps

from uniface.headpose import HeadPose
from uniface.onnx_utils import create_onnx_session
from uniface.parsing import BiSeNet
from uniface.quality import EDifFIQA

from utils.face_pipeline import (FaceFrame, FaceRecord, FaceTracker,
                                 TrackAttributeVoter, output_layout,
                                 render_face_frame)
from utils.face_utils import (align_face, detect_face, estimate_gaze,
                              estimate_head_pose, parse_face,
                              predict_age_gender, predict_emotion)
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


class GpuParsingFallback:
    """Retry the known OpenVINO GPU primitive collision on OpenVINO CPU."""

    def __init__(self, parser):
        self.parser = parser
        self.fallback_providers = [('OpenVINOExecutionProvider',
                                    {'device_type': 'CPU'})]

    @property
    def session(self):
        return self.parser.session

    def parse(self, image):
        try:
            return self.parser.parse(image)
        except Exception as error:
            message = str(error)
            if (self.fallback_providers is None
                    or '[GPU] Different primitive with id' not in message
                    or self.session.get_providers()[0] != 'OpenVINOExecutionProvider'):
                raise
            self.parser.session = create_onnx_session(
                self.parser.model_path, providers=self.fallback_providers)
            verify_model_providers({'parsing': self.parser}, self.fallback_providers)
            self.fallback_providers = None
            print('[WARN] OpenVINO GPU face parsing hit a primitive-ID collision; '
                  'using OpenVINO CPU for face parsing. Other models remain on GPU.')
            return self.parser.parse(image)


def openvino_parser_session(model_path, providers):
    """Use OpenVINO's recommended ORT graph setting for face parsing."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(model_path, sess_options=options,
                                providers=providers)


class FaceDetectionPipeline:
    """Reusable FD processor. process_frame returns data independent of rendering."""

    def __init__(self, detector, age_gender, emotion_predictor, gaze_estimator,
                 parser=None, confidence_threshold=0.6, infer_interval=5,
                 tracking=False, face_states=None, quality_estimator=None,
                 vote_window=5, head_pose_estimator=None):
        self.detector = detector
        self.age_gender = age_gender
        self.emotion_predictor = emotion_predictor
        self.gaze_estimator = gaze_estimator
        self.parser = parser
        self.face_states = face_states
        self.quality_estimator = quality_estimator
        self.head_pose_estimator = head_pose_estimator
        self.confidence_threshold = confidence_threshold
        if infer_interval < 1:
            raise ValueError('infer_interval must be at least 1.')
        self.infer_interval = infer_interval
        self.tracker = FaceTracker() if tracking else None
        self.voter = TrackAttributeVoter(vote_window) if tracking else None
        self.last = FaceFrame(-1)

    def process_frame(self, frame, frame_index):
        inferred = frame_index % self.infer_interval == 0 or self.last.frame_index < 0
        if inferred:
            start = time.perf_counter()
            faces = detect_face(frame, self.detector,
                                confidence_threshold=self.confidence_threshold)
            ages, genders = (predict_age_gender(frame, self.age_gender, faces)
                             if self.age_gender is not None else
                             ([None] * len(faces), [None] * len(faces)))
            emotions = (predict_emotion(frame, self.emotion_predictor, faces)
                        if self.emotion_predictor else ['N/A'] * len(faces))
            pitches, yaws = (estimate_gaze(frame, self.gaze_estimator, faces)
                            if self.gaze_estimator is not None else
                            ([None] * len(faces), [None] * len(faces)))
            aligned = align_face(frame, faces)
            if self.parser:
                parsing_maps, masked_crops = parse_face(frame, self.parser, aligned)
                parsing_visuals = [vis_parsing_maps(crop, class_map)
                                   for crop, class_map in zip(aligned, parsing_maps)]
            else:
                parsing_maps = masked_crops = [None] * len(faces)
                parsing_visuals = [None] * len(faces)
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
                if self.head_pose_estimator is not None:
                    pose = estimate_head_pose(frame, self.head_pose_estimator, face)
                    if pose is not None:
                        attributes.update(head_pitch=pose.pitch, head_yaw=pose.yaw,
                                          head_roll=pose.roll)
                records.append(FaceRecord(
                    bbox=tuple(face.bbox), landmarks=np.asarray(face.landmarks),
                    confidence=float(face.confidence), crop=aligned[index],
                    mask=masked_crops[index], attributes=attributes,
                    extras=({'parsing_map': parsing_maps[index],
                             'parsing_visualization': parsing_visuals[index]}
                            if self.parser else {})))
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
    parser.add_argument('-o', '--output-dir', default='./output', help='Output video directory')
    parser.add_argument('-I', '--infer-frame-interval', type=int, default=5,
                        help='Run detection every N frames')
    parser.add_argument('--confidence-threshold', type=float, default=0.6)
    parser.add_argument('--detector-input-size', type=int, default=640)
    parser.add_argument('--parse-face', action='store_true',
                        help='Show original and UniFace-parsed face crops in both layouts (default: off)')
    parser.add_argument('--age-gender', action='store_true',
                        help='Estimate age and gender (default: off)')
    parser.add_argument('--emotion', action='store_true',
                        help='Estimate emotion (default: off)')
    parser.add_argument('--no-emotion', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--face-states', action='store_true',
                        help='Predict eye, glasses, sunglasses, and mask states')
    parser.add_argument('--face-quality', action='store_true',
                        help='Score aligned face image quality')
    parser.add_argument('--head-pose', action='store_true',
                        help='Estimate head pitch, yaw, and roll (requires head-pose weight)')
    parser.add_argument('--gaze', action='store_true',
                        help='Estimate gaze and draw its arrow (default: off)')
    parser.add_argument('--show-landmarks', action='store_true',
                        help='Draw five detector landmarks (default: off)')
    parser.add_argument('--result-layout', choices=('panel', 'box'), default='box',
                        help='Attached panel or frame-only cards beside faces')
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
    if args.infer_frame_interval < 1:
        raise SystemExit('--infer_frame_interval must be at least 1.')
    interval = args.infer_frame_interval
    threshold = min(max(args.confidence_threshold, 0.0), 1.0)
    providers = resolve_providers(args.device, args.openvino_device,
                                  args.openvino_precision)
    if args.device == 'cuda' and args.emotion and not args.no_emotion and not torch.cuda.is_available():
        raise RuntimeError('CUDA was selected, but PyTorch CUDA is unavailable for emotion inference.')
    with uniface_providers(providers):
        detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34,
                              confidence_threshold=0.5, nms_threshold=0.4,
                              input_size=(args.detector_input_size, args.detector_input_size),
                              **model_provider_kwargs(RetinaFace, providers))
        verify_model_providers({'detector': detector}, providers)
        age_gender = (AgeGender(**model_provider_kwargs(AgeGender, providers))
                      if args.age_gender else None)
        gaze = (MobileGaze(model_name=GazeWeights.RESNET18,
                           **model_provider_kwargs(MobileGaze, providers))
                if args.gaze else None)
        parser_kwargs = {'input_size': (112, 112)} if 'input_size' in signature(BiSeNet).parameters else {}
        face_parser = (BiSeNet(model_name=ParsingWeights.RESNET34,
                              **parser_kwargs, **model_provider_kwargs(BiSeNet, providers))
                       if args.parse_face else None)
        if face_parser is not None and args.device == 'openvino' and args.openvino_device == 'GPU':
            face_parser.session = openvino_parser_session(face_parser.model_path, providers)
            face_parser = GpuParsingFallback(face_parser)
        face_states = (FaceAttribNet(**model_provider_kwargs(FaceAttribNet, providers))
                       if args.face_states else None)
        quality_estimator = (EDifFIQA(model_name=EDifFIQAWeights.T,
                                     **model_provider_kwargs(EDifFIQA, providers))
                             if args.face_quality else None)
        head_pose = None
        if args.head_pose:
            try:
                head_pose = HeadPose(
                    model_name=HeadPoseWeights.MOBILENET_V3_SMALL,
                    **model_provider_kwargs(HeadPose, providers))
            except ConnectionError as error:
                raise RuntimeError(
                    'Head pose needs headpose_mobilenetv3_small.onnx in the '
                    'UniFace model cache. Add it to models/uniface and set '
                    'UNIFACE_CACHE_DIR=./models/uniface, or restore model downloads.'
                ) from error
    models = {'detector': detector}
    if age_gender is not None:
        models['age_gender'] = age_gender
    if gaze is not None:
        models['gaze'] = gaze
    if face_parser is not None:
        models['parsing'] = face_parser
    if face_states is not None:
        models['face_states'] = face_states
    if quality_estimator is not None:
        models['face_quality'] = quality_estimator
    if head_pose is not None:
        models['head_pose'] = head_pose
    verify_model_providers(models, providers)
    emotion = None
    if args.emotion and not args.no_emotion:
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
        layout = output_layout((height, width, 3),
                               min_side_width=1536 if args.parse_face and
                               args.result_layout == 'panel' else 720,
                               result_layout=args.result_layout)
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
            vote_window=args.track_vote_window, head_pose_estimator=head_pose)
        progress = tqdm(total=total)
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = pipeline.process_frame(frame, frame_index)
                canvas = render_face_frame(frame, result, 'FD', layout,
                                           infer_interval=interval,
                                           result_layout=args.result_layout,
                                           show_gaze=args.gaze,
                                           show_landmarks=args.show_landmarks,
                                           head_pose_enabled=args.head_pose)
                writer.write(canvas)
                if not args.no_display:
                    preview = canvas
                    if args.display_width > 0 and canvas.shape[1] > args.display_width:
                        preview_height = int(args.display_width * canvas.shape[0] / canvas.shape[1])
                        preview = cv2.resize(canvas, (args.display_width, preview_height))
                    cv2.imshow(os.path.basename(input_path), preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        return
                    if key == 27:
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
