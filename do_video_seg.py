#!/usr/bin/env python3
"""Run FastSAM or YOLOE instance segmentation on video and save an MP4."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import onnxruntime as ort
import torch

from utils.inference_device import resolve_providers
from utils.utils import get_available_devices, parse_video_device, put_text_to_canvas
from utils.video_writer import EncodedVideoWriter, add_encoding_arguments


MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'ultralytics'
BOTTOM_INFO_HEIGHT = 84
DEFAULT_SEG_WEIGHTS = {
    'fastsam': 'FastSAM-s.pt',
    'yoloe-prompt-free': 'yoloe-11s-seg-pf.pt',
    'yoloe-text': 'yoloe-11s-seg.pt',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--list-devices', action='store_true',
                        help='List available camera devices')
    parser.add_argument('-i', '--input-device', '--input_device', default=None,
                        help='Video file, camera index, or stream URL')
    parser.add_argument('-y', '--yt-url', '--YT_URL', action='store_true',
                        help='Resolve a YouTube URL before opening it')
    parser.add_argument('-r', '--resize-ratio', '--resize_ratio', type=float, default=1.0,
                        help='Resize frames before inference and encoding (default: 1)')
    parser.add_argument('-m', '--mask-only', '--mask_only', action='store_true',
                        help='Draw colored masks on black instead of the video frame')
    parser.add_argument('-s', '--skip-frame', '--skip_frame', '--infer-frame-interval',
                        type=int, default=3,
                        help='Run models on frames 0, N, 2N, ... (default: 3)')
    parser.add_argument('-o', '--option', choices=('seg-cls', 'seg', 'cls'),
                        default='seg', help='Analysis to run (default: seg)')
    parser.add_argument('--seg-model', choices=('fastsam', 'yoloe'), default='fastsam',
                        help='Segmentation model (default: fastsam)')
    parser.add_argument('--seg-weights', default=None,
                        help='Custom segmentation .pt or exported .onnx file')
    parser.add_argument('--cls-weights', default=str(MODEL_DIR / 'yolo11n-cls.pt'),
                        help='Classification model weights')
    parser.add_argument('--prompt', action='append', default=[], metavar='TEXT',
                        help='YOLOE text class; repeat for multiple classes')
    parser.add_argument('--prompt-free', action='store_true',
                        help='Use YOLOE built-in vocabulary (default when no --prompt)')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'openvino'),
                        default='auto', help='Inference device (default: auto)')
    parser.add_argument('--openvino-device', choices=('CPU', 'GPU'), default='CPU',
                        help='OpenVINO target (default: CPU)')
    parser.add_argument('--openvino-precision', choices=('FP32', 'FP16'), default='FP32',
                        help='OpenVINO export precision (default: FP32)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Model input image size (default: 640)')
    parser.add_argument('--confidence', type=float, default=0.25,
                        help='Segmentation confidence threshold (default: 0.25)')
    parser.add_argument('--border-width', type=int, default=5,
                        help='Black segmentation boundary thickness (default: 5)')
    parser.add_argument('--min-segment-area', type=int, default=200, metavar='PIXELS',
                        help='Discard connected mask fragments smaller than this area (default: 200; 0 disables)')
    parser.add_argument('--mask-opacity', type=float, default=0.2,
                        help='Colored mask opacity, from 0 to 1 (default: 0.2)')
    parser.add_argument('--output', default=None, help='Output MP4 path')
    parser.add_argument('--output-dir', default='./output',
                        help='Output directory when --output is omitted')
    parser.add_argument('--no-output', action='store_true',
                        help='Disable MP4 writing')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable the live preview window')
    add_encoding_arguments(parser)
    args = parser.parse_args(argv)
    if args.list_devices:
        return args
    if args.skip_frame < 1:
        parser.error('--skip-frame must be at least 1')
    if not math.isfinite(args.resize_ratio) or args.resize_ratio <= 0:
        parser.error('--resize-ratio must be positive')
    if args.imgsz < 32:
        parser.error('--imgsz must be at least 32')
    if not 0 <= args.confidence <= 1:
        parser.error('--confidence must be between 0 and 1')
    if args.border_width < 1:
        parser.error('--border-width must be at least 1')
    if args.min_segment_area < 0:
        parser.error('--min-segment-area must be nonnegative')
    if not math.isfinite(args.mask_opacity) or not 0 <= args.mask_opacity <= 1:
        parser.error('--mask-opacity must be between 0 and 1')
    args.prompt = [value.strip() for value in args.prompt]
    if any(not value for value in args.prompt):
        parser.error('--prompt cannot be empty')
    if args.prompt and (args.seg_model != 'yoloe' or args.option == 'cls'):
        parser.error('--prompt requires --seg-model yoloe and segmentation enabled')
    if args.prompt and args.prompt_free:
        parser.error('--prompt and --prompt-free cannot be combined')
    if args.prompt_free and (args.seg_model != 'yoloe' or args.option == 'cls'):
        parser.error('--prompt-free requires --seg-model yoloe and segmentation enabled')
    if args.seg_model == 'yoloe' and args.seg_weights:
        name = Path(args.seg_weights).name
        if args.prompt and '-seg-pf' in name:
            parser.error('Text prompts require a YOLOE *-seg.pt checkpoint, not *-seg-pf.pt')
        if args.prompt_free and name.endswith('-seg.pt'):
            parser.error('Prompt-free mode requires a YOLOE *-seg-pf.pt checkpoint')
    if args.device != 'openvino' and (args.openvino_device != 'CPU' or
                                      args.openvino_precision != 'FP32'):
        parser.error('OpenVINO target and precision require --device openvino')
    if args.openvino_precision == 'FP16' and args.openvino_device != 'GPU':
        parser.error('OpenVINO FP16 requires --openvino-device GPU')
    if args.output and args.no_output:
        parser.error('--output and --no-output cannot be combined')
    return args


def resolve_device(args):
    device = args.device
    if device == 'auto':
        device = ('cuda' if torch.cuda.is_available() else
                  'openvino' if 'OpenVINOExecutionProvider' in ort.get_available_providers()
                  else 'cpu')
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested, but PyTorch cannot see a CUDA GPU.')
        return device, '0'
    if device == 'openvino':
        resolve_providers('openvino', args.openvino_device,
                          args.openvino_precision)
        # Ultralytics preprocesses on CPU; ONNX Runtime sends model nodes to OpenVINO.
        return device, 'cpu'
    return device, 'cpu'


def configure_openvino_backend(providers):
    """Run Ultralytics ONNX models through the requested OpenVINO EP."""
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.nn.backends.onnx import ONNXBackend

    class OpenVINOONNXBackend(ONNXBackend):
        def load_model(self, weight):
            self.session = ort.InferenceSession(str(weight), providers=providers)
            requested = (providers[0][0] if isinstance(providers[0], tuple)
                         else providers[0])
            if self.session.get_providers()[0] != requested:
                raise RuntimeError(f'OpenVINO requested, but {weight} uses '
                                   f'{self.session.get_providers()}')
            self.output_names = [item.name for item in self.session.get_outputs()]
            metadata = self.session.get_modelmeta().custom_metadata_map
            if metadata:
                self.apply_metadata(dict(metadata))
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = 'float16' in self.session.get_inputs()[0].type
            self.use_io_binding = False
            print(f'[INFO] {Path(weight).name}: {self.session.get_providers()[0]} '
                  f'({providers[0]})')

    AutoBackend._BACKEND_MAP['onnx'] = OpenVINOONNXBackend


def _ultralytics_models():
    # Keep Ultralytics settings inside the project when the user has no config path.
    if 'YOLO_CONFIG_DIR' not in os.environ:
        config_dir = Path(__file__).resolve().parent / 'output' / 'ultralytics_config'
        (config_dir / 'Ultralytics').mkdir(parents=True, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    from ultralytics import FastSAM, YOLO, YOLOE
    return FastSAM, YOLO, YOLOE


def _default_weights(name):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODEL_DIR / DEFAULT_SEG_WEIGHTS[name])


def _onnx_cache_path(weights, prompts, imgsz):
    path = Path(weights)
    stat = path.stat() if path.is_file() else None
    key = json.dumps({
        'weights': str(path.resolve()),
        'size': stat.st_size if stat else None,
        'modified_ns': stat.st_mtime_ns if stat else None,
        'prompts': prompts or [],
        'imgsz': imgsz,
    }, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    cache_dir = MODEL_DIR / 'exports'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{path.stem}_{digest}.onnx'


@contextmanager
def _model_download_dir():
    """Place YOLOE's lazily downloaded text encoder beside the other weights."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(MODEL_DIR)
        yield
    finally:
        os.chdir(previous)


def _load_model(weights, model_class, device, args, prompts=None,
                imgsz=None, task=None):
    _, YOLO, _ = _ultralytics_models()
    imgsz = args.imgsz if imgsz is None else imgsz
    path = Path(weights)
    if weights == args.seg_weights and not path.exists():
        raise FileNotFoundError(f'Model weights not found: {weights}')
    if path.is_dir():
        raise ValueError('Pass a .pt checkpoint or an exported .onnx file. '
                         'OpenVINO IR directories are not used by this script.')
    if path.suffix == '.onnx':
        if device != 'openvino':
            raise ValueError('Exported ONNX weights require --device openvino '
                             'in this environment.')
        if prompts:
            raise ValueError('YOLOE prompts cannot be changed in an exported model; '
                             'pass the original .pt weights instead.')
        return YOLO(str(path), task=task)
    cache_path = (_onnx_cache_path(weights, prompts, imgsz)
                  if device == 'openvino' and path.is_file() else None)
    if cache_path is not None and cache_path.is_file():
        print(f'[INFO] Reusing OpenVINO ONNX export: {cache_path}')
        return YOLO(str(cache_path), task=task)
    try:
        model = model_class(weights)
    except ConnectionError as error:
        raise RuntimeError(
            f'Cannot download model weights for {weights}. Add the checkpoint '
            f'to {MODEL_DIR} or pass --seg-weights with a local file.') from error
    if prompts:
        try:
            with _model_download_dir():
                model.set_classes(prompts)
        except ConnectionError as error:
            raise RuntimeError(f'YOLOE text prompts need MobileCLIP at '
                               f'{MODEL_DIR / "mobileclip_blt.ts"}. Add that file, '
                               'restore network access, or use --prompt-free.') from error
    if device == 'openvino':
        if cache_path is None:
            cache_path = _onnx_cache_path(weights, prompts, imgsz)
        print(f'[INFO] Exporting {weights} to ONNX for OpenVINO EP; '
              'prompts, if any, are baked into this export.')
        model.model.pt_path = str(cache_path.with_suffix('.pt'))
        exported = model.export(format='onnx', imgsz=imgsz,
                                simplify=False, device='cpu')
        return YOLO(exported, task=task)
    return model


def load_models(args, device):
    FastSAM, YOLO, YOLOE = _ultralytics_models()
    seg_model = cls_model = None
    if args.option != 'cls':
        if args.seg_model == 'yoloe':
            mode = 'text' if args.prompt else 'prompt-free'
            weights = (args.seg_weights or
                       _default_weights(f'yoloe-{mode}'))
            seg_model = _load_model(weights, YOLOE, device, args,
                                    prompts=args.prompt, task='segment')
            print(f'[INFO] YOLOE mode: {mode}; weights: {weights}')
        else:
            weights = args.seg_weights or _default_weights('fastsam')
            seg_model = _load_model(weights, FastSAM, device, args,
                                    task='segment')
            print(f'[INFO] FastSAM weights: {weights}')
    if args.option != 'seg':
        cls_model = _load_model(args.cls_weights, YOLO, device, args,
                                imgsz=224, task='classify')
    return seg_model, cls_model


def _mask_color(index):
    hsv = np.uint8([[[int(index * 47) % 180, 210, 245]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def render_segmentation(image, result, mask_only=False, border_width=5,
                        min_segment_area=200, mask_opacity=0.8):
    """Overlay instance masks and black contours at the source frame size."""
    canvas = np.zeros_like(image) if mask_only else image.copy()
    if result is None or result.masks is None:
        return canvas
    mask_data = result.masks.data
    if hasattr(mask_data, 'cpu'):
        mask_data = mask_data.cpu().numpy()
    masks = np.asarray(mask_data)
    height, width = image.shape[:2]
    contours_to_draw = []
    for index, mask in enumerate(masks):
        binary = (mask > 0.5).astype(np.uint8)
        if binary.shape != (height, width):
            binary = cv2.resize(binary, (width, height),
                                interpolation=cv2.INTER_NEAREST)
        if min_segment_area > 1:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8)
            keep = np.zeros(count, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_segment_area
            binary = keep[labels].astype(np.uint8)
        if not np.any(binary):
            continue
        color = np.asarray(_mask_color(index), dtype=np.float32)
        selected = binary.astype(bool)
        canvas[selected] = np.rint(canvas[selected] * (1 - mask_opacity) +
                                   color * mask_opacity).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours_to_draw.extend(contours)
    if contours_to_draw:
        cv2.drawContours(canvas, contours_to_draw, -1, (0, 0, 0), border_width)
    return canvas


def _fitted_text(canvas, value, y):
    available = canvas.shape[1] - 20
    if available <= 0:
        return
    scale = 0.5
    full_text = value
    while value and cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > available:
        value = value[:-1]
    if value != full_text and len(value) > 3:
        value = value[:-3] + '...'
    if value:
        cv2.putText(canvas, value, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (235, 240, 245), 1, cv2.LINE_AA)


def add_model_info_strip(canvas, args, device):
    height, width = canvas.shape[:2]
    result = np.full((height + BOTTOM_INFO_HEIGHT, width, 3), (20, 25, 34),
                     dtype=np.uint8)
    result[:height] = canvas
    cv2.line(result, (0, height), (width - 1, height), (100, 110, 125), 1)
    if args.option == 'cls':
        model_name = f'Classifier: {Path(args.cls_weights).name}'
    else:
        weight = args.seg_weights or _default_weights(
            'fastsam' if args.seg_model == 'fastsam' else
            'yoloe-text' if args.prompt else 'yoloe-prompt-free')
        model_name = f'Model: {Path(weight).name}'
        if args.option == 'seg-cls':
            model_name += f' | Classifier: {Path(args.cls_weights).name}'
    target = (f'OpenVINO {args.openvino_device} {args.openvino_precision}'
              if device == 'openvino' else device.upper())
    _fitted_text(result, f'{model_name} | Device: {target}', height + 25)
    prompt = (', '.join(args.prompt) if args.prompt else 'prompt-free'
              if args.seg_model == 'yoloe' and args.option != 'cls' else 'none')
    _fitted_text(result, f'Prompt: {prompt}', height + 49)
    _fitted_text(result, f'Min area: {args.min_segment_area} px | '
                 f'Mask: {args.mask_opacity:.2f} | Border: {args.border_width} px',
                 height + 73)
    return result


def _output_path(args, input_device):
    if args.output:
        path = Path(args.output)
    else:
        if isinstance(input_device, int):
            stem = f'camera_{input_device}_{datetime.now():%Y%m%d_%H%M%S}'
        elif Path(str(input_device)).is_file():
            stem = Path(input_device).stem
        else:
            stem = urlparse(str(input_device)).hostname or 'stream'
            stem = f'{stem}_{datetime.now():%Y%m%d_%H%M%S}'
        path = Path(args.output_dir) / f'{stem}_output_{args.option.replace("-", "_")}.mp4'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    args = parse_args(argv)
    if args.list_devices:
        print(f'[INFO] Found devices: {get_available_devices(max_index=10)}')
        return
    device, predict_device = resolve_device(args)
    if device == 'openvino':
        _ultralytics_models()
        configure_openvino_backend(resolve_providers(
            'openvino', args.openvino_device, args.openvino_precision))
    input_spec = args.input_device
    if input_spec is None:
        cameras = get_available_devices(number_of_devices=1, max_index=10)
        if not cameras:
            raise RuntimeError('No camera was found; pass --input-device with a video path.')
        input_spec = str(cameras[0])
    input_device = parse_video_device(input_spec, YT_URL=args.yt_url)
    if (args.output and isinstance(input_device, str) and
            Path(input_device).is_file() and
            Path(args.output).resolve() == Path(input_device).resolve()):
        raise ValueError('Output video must not overwrite the input video.')
    cap = cv2.VideoCapture(input_device)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video input: {input_device}')
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        seg_model, cls_model = load_models(args, device)
        if device == 'openvino':
            print(f'[INFO] Inference device: OpenVINO EP target '
                  f'{args.openvino_device}, precision hint '
                  f'{args.openvino_precision}; preprocessing on CPU')
        else:
            print(f'[INFO] Inference device: {device} ({predict_device})')
        last_seg = last_cls = None
        last_infer_fps = 0.0
        frame_index = 0
        show_osd = True
        mask_only = args.mask_only
        interval = args.skip_frame
        input_is_file = isinstance(input_device, str) and Path(input_device).is_file()
        failures = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                if input_is_file or failures >= 29:
                    break
                failures += 1
                time.sleep(0.1)
                continue
            failures = 0
            if args.resize_ratio != 1:
                frame = cv2.resize(frame, None, fx=args.resize_ratio,
                                   fy=args.resize_ratio, interpolation=cv2.INTER_AREA)
            if writer is None and not args.no_output:
                path = _output_path(args, input_device)
                writer = EncodedVideoWriter(path, (frame.shape[1], frame.shape[0] + BOTTOM_INFO_HEIGHT),
                                            fps, args.video_codec, args.video_crf,
                                            args.video_preset)
                print(f'[INFO] OUTPUT: {path} '
                      f'({frame.shape[1]}x{frame.shape[0] + BOTTOM_INFO_HEIGHT}@{fps:.2f})')
            if frame_index % interval == 0:
                start = time.perf_counter()
                if seg_model is not None:
                    last_seg = seg_model.predict(frame, device=predict_device,
                                                 imgsz=args.imgsz,
                                                 conf=args.confidence,
                                                 retina_masks=True,
                                                 verbose=False)[0]
                if cls_model is not None:
                    last_cls = cls_model.predict(frame, device=predict_device,
                                                 imgsz=224, verbose=False)[0]
                elapsed = time.perf_counter() - start
                last_infer_fps = 1 / elapsed if elapsed > 0 else 0.0
            canvas = (render_segmentation(frame, last_seg, mask_only,
                                          args.border_width,
                                          args.min_segment_area,
                                          args.mask_opacity)
                      if seg_model is not None else frame.copy())
            if last_cls is not None:
                canvas = np.ascontiguousarray(
                    last_cls.plot(img=canvas, boxes=False, masks=False,
                                  labels=False, probs=True)).copy()
            if show_osd:
                status = (f'{args.seg_model if seg_model else "cls"}  '
                          f'frame {frame_index}  infer {last_infer_fps:.1f} FPS  '
                          f'every {interval} frame')
                cv2.rectangle(canvas, (0, 0),
                              (canvas.shape[1] - 1, min(40, canvas.shape[0] - 1)),
                              (20, 25, 34), -1)
                put_text_to_canvas(canvas, status, top_left=(10, 30),
                                   font_scale=0.5, fg_color=(245, 245, 245),
                                   thickness=1)
            canvas = add_model_info_strip(canvas, args, device)
            if writer is not None:
                writer.write(canvas)
            if not args.no_display:
                cv2.imshow(f'Segmentation: {input_device}', canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key in (ord('='), ord('+')):
                    interval += 1
                elif key == ord('-'):
                    interval = max(1, interval - 1)
                elif key == ord('m'):
                    mask_only = not mask_only
                elif key == 13:
                    show_osd = not show_osd
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
