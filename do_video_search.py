#!/usr/bin/env python3
"""Search video frames with prompted detectors, LocateAnything, or an OpenAI VLM."""

import argparse
import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from utils.inference_device import resolve_providers
from utils.utils import get_available_devices, parse_video_device
from utils.video_writer import EncodedVideoWriter, add_encoding_arguments


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models' / 'video_search'
ULTRA_DIR = ROOT / 'models' / 'ultralytics'
PANEL_WIDTH = 320
STRIP_HEIGHT = 94


@dataclass
class Hit:
    label: str
    score: float | None = None
    box: tuple[int, int, int, int] | None = None
    description: str = ''


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input-device', default='0',
                        help='Video path, stream URL, or webcam index (default: 0)')
    parser.add_argument('-l', '--list-devices', action='store_true')
    parser.add_argument('--backend', choices=('owlvit', 'yoloe', 'yolo-world',
                                              'locateanything', 'openai'),
                        default='owlvit')
    parser.add_argument('--prompt', action='append', default=[], metavar='TEXT',
                        help='Target object or question; repeat for multiple targets')
    parser.add_argument('--weights', default=None,
                        help='Local model directory or checkpoint (local backends)')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'openvino'),
                        default='auto')
    parser.add_argument('--precision', choices=('FP32', 'FP16'), default='FP32')
    parser.add_argument('--openvino-device', choices=('CPU', 'GPU'), default='CPU')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Maximum search image side in pixels (default: 640)')
    parser.add_argument('--infer-frame-interval', '-s', type=int, default=1,
                        help='Run search on frames 0, N, 2N, ... (default: 1)')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--prefilter', choices=('none', 'person'), default='none',
                        help='Search expanded person crops instead of whole frames')
    parser.add_argument('--prefilter-weights', default=str(ULTRA_DIR / 'yolo11n.pt'))
    parser.add_argument('--prefilter-confidence', type=float, default=0.35)
    parser.add_argument('--crop-expand', type=float, default=0.25,
                        help='Expand each person box by this fraction on each side (default: 0.25)')
    parser.add_argument('--show-prefilter', action='store_true',
                        help='Draw original person boxes and expanded search crops')
    parser.add_argument('--vlm-mode', choices=('image', 'clip'), default='image',
                        help='VLM input: one image or recent video frames')
    parser.add_argument('--vlm-prompt-mode', choices=('separate', 'combine'),
                        default='separate',
                        help='Ask each VLM prompt separately, or combine them in one query')
    parser.add_argument('--clip-frames', type=int, default=4,
                        help='Number of recent frames per clip request (default: 4)')
    parser.add_argument('--vlm-provider', choices=('llamacpp', 'openai'),
                        default='llamacpp', help='VLM API provider (default: llamacpp)')
    parser.add_argument('--vlm-base-url', default=None,
                        help='OpenAI-compatible API base URL; default is the local llama.cpp server')
    parser.add_argument('--vlm-model', '--openai-model', default='auto',
                        help='VLM model ID; auto queries local /v1/models')
    parser.add_argument('--vlm-timeout', '--openai-timeout', type=float, default=180,
                        help='VLM request timeout in seconds (default: 180)')
    parser.add_argument('--alert', action='store_true',
                        help='Draw a thick red frame border after repeated hits')
    parser.add_argument('--alert-after', type=int, default=3,
                        help='Consecutive inference results before alert (default: 3)')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Stop after N input frames; 0 means no limit')
    parser.add_argument('--output', default=None)
    parser.add_argument('--output-dir', default=str(ROOT / 'output'))
    parser.add_argument('--no-output', action='store_true')
    parser.add_argument('--no-display', action='store_true')
    add_encoding_arguments(parser)
    args = parser.parse_args(argv)
    if args.list_devices:
        return args
    args.prompt = [value.strip() for value in args.prompt]
    if not args.prompt or any(not value for value in args.prompt):
        parser.error('provide at least one nonempty --prompt')
    if args.imgsz < 32 or args.infer_frame_interval < 1 or args.clip_frames < 1:
        parser.error('--imgsz must be >=32; interval and clip frames must be >=1')
    if args.alert_after < 1 or args.max_frames < 0:
        parser.error('--alert-after must be >=1; --max-frames must be >=0')
    if not 0 <= args.confidence <= 1 or not 0 <= args.prefilter_confidence <= 1:
        parser.error('confidence values must be between 0 and 1')
    if not math.isfinite(args.crop_expand) or args.crop_expand < 0:
        parser.error('--crop-expand must be nonnegative')
    if args.show_prefilter and args.prefilter == 'none':
        parser.error('--show-prefilter requires --prefilter person')
    if not math.isfinite(args.vlm_timeout) or args.vlm_timeout <= 0:
        parser.error('--vlm-timeout must be positive')
    if args.output and args.no_output:
        parser.error('--output and --no-output cannot be combined')
    if args.backend != 'openai' and args.vlm_mode != 'image':
        parser.error('--vlm-mode clip requires --backend openai')
    return args


def resolve_device(args):
    if args.backend == 'openai':
        # Device flags still select an optional local person prefilter. The VLM
        # itself runs remotely and has no local precision setting.
        if args.prefilter == 'none':
            if args.device != 'auto' or args.precision != 'FP32':
                raise ValueError('OpenAI VLM runs remotely; device and precision '
                                 'only apply when --prefilter person is enabled.')
            return 'remote'
    if args.backend == 'locateanything' and args.device == 'openvino':
        raise RuntimeError('LocateAnything does not support OpenVINO export.')
    device = args.device
    if device == 'auto':
        import torch
        device = ('cuda' if torch.cuda.is_available() else
                  'openvino' if (args.backend in ('owlvit', 'yoloe', 'yolo-world') or
                                 (args.backend == 'openai' and args.prefilter == 'person')) and
                  _openvino_available() else 'cpu')
    if device == 'openvino':
        if args.backend == 'locateanything':
            raise RuntimeError(f'{args.backend} has no validated OpenVINO path in this script.')
        resolve_providers('openvino', args.openvino_device, args.precision)
    elif device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but PyTorch cannot see a CUDA GPU.')
    elif args.openvino_device != 'CPU':
        raise ValueError('--openvino-device requires --device openvino')
    if args.precision == 'FP16' and device == 'cpu':
        raise ValueError('FP16 requires CUDA or OpenVINO GPU.')
    if args.backend == 'locateanything' and args.precision == 'FP16':
        raise RuntimeError('LocateAnything precision is controlled by its model loader; FP16 is unavailable here.')
    return device


def _openvino_available():
    import onnxruntime as ort
    return 'OpenVINOExecutionProvider' in ort.get_available_providers()


def _max_side(image, size):
    h, w = image.shape[:2]
    if max(h, w) <= size:
        return image, 1.0, 1.0
    ratio = size / max(h, w)
    resized = cv2.resize(image, (max(1, round(w * ratio)), max(1, round(h * ratio))),
                         interpolation=cv2.INTER_AREA)
    return resized, w / resized.shape[1], h / resized.shape[0]


def _boxes_from_ultralytics(result, names, offset=(0, 0), scale=(1, 1)):
    hits = []
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        coords = tuple(round(xyxy[i] * scale[i % 2] + offset[i % 2]) for i in range(4))
        label = names[int(box.cls[0])]
        hits.append(Hit(str(label), float(box.conf[0]), coords))
    return hits


def _cache_path(weights, prompts, imgsz):
    path = Path(weights)
    stat = path.stat()
    key = json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                      prompts, imgsz], sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    folder = MODEL_DIR / 'exports'
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{path.stem}_{digest}.onnx'


def _ultralytics_imports():
    # Ultralytics checks its config directory at import time.
    config = ROOT / 'output' / 'ultralytics_config'
    (config / 'Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(config))
    from ultralytics import YOLO, YOLOE, YOLOWorld
    return YOLO, YOLOE, YOLOWorld


def _load_yolo(weights, kind, prompts, device, args):
    YOLO, YOLOE, YOLOWorld = _ultralytics_imports()
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        if path.parent == ULTRA_DIR and path.name in (
                'yoloe-11s-seg.pt', 'yolov8s-worldv2.pt', 'yolo11n.pt'):
            from ultralytics.utils.downloads import attempt_download_asset
            ULTRA_DIR.mkdir(parents=True, exist_ok=True)
            try:
                attempt_download_asset(str(path))
            except ConnectionError as error:
                raise FileNotFoundError(
                    f'Model weights missing: {path}. Download failed; add the '
                    'checkpoint there or restore network access.') from error
        if not path.is_file():
            raise FileNotFoundError(f'Model weights missing: {path}')
    task = 'segment' if kind == 'yoloe' else 'detect'
    if device == 'openvino':
        from do_video_seg import configure_openvino_backend
        configure_openvino_backend(resolve_providers('openvino', args.openvino_device,
                                                     args.precision))
        export = _cache_path(path, prompts, args.imgsz)
        if export.is_file():
            return YOLO(str(export), task=task)
    if kind == 'yoloe':
        model = YOLOE(str(path))
    elif kind == 'yolo-world':
        if device == 'openvino' and 'worldv2' not in path.name:
            raise RuntimeError('YOLO-World OpenVINO export requires a worldv2 checkpoint.')
        model = YOLOWorld(str(path))
    else:
        model = YOLO(str(path))
    if prompts:
        old_cwd = Path.cwd()
        try:
            os.chdir(ULTRA_DIR)
            try:
                model.set_classes(prompts)
            except ConnectionError as error:
                raise RuntimeError('Text prompting needs a local encoder; for '
                                   'YOLOE-11 place mobileclip_blt.ts in '
                                   f'{ULTRA_DIR}.') from error
        finally:
            os.chdir(old_cwd)
    if device == 'openvino':
        model.model.pt_path = str(export.with_suffix('.pt'))
        exported = model.export(format='onnx', imgsz=args.imgsz,
                                simplify=False, device='cpu')
        return YOLO(exported, task=task)
    return model


class YoloSearch:
    def __init__(self, args, device):
        defaults = {'yoloe': ULTRA_DIR / 'yoloe-11s-seg.pt',
                    'yolo-world': ULTRA_DIR / 'yolov8s-worldv2.pt'}
        self.name = args.backend
        self.device = device
        self.args = args
        self.model = _load_yolo(args.weights or defaults[args.backend], args.backend,
                                args.prompt, device, args)

    def search(self, image):
        result = self.model.predict(image, device='cpu' if self.device == 'openvino'
                                    else '0' if self.device == 'cuda' else 'cpu',
                                    imgsz=self.args.imgsz, conf=self.args.confidence,
                                    half=self.device == 'cuda' and self.args.precision == 'FP16',
                                    verbose=False)[0]
        return _boxes_from_ultralytics(result, result.names)


class OwlSearch:
    def __init__(self, args, device):
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        self.name = 'OWL-ViT'
        self.args = args
        self.device = device
        path = Path(args.weights or ROOT / 'models/google/owlvit-base-patch32')
        if not path.is_dir():
            raise FileNotFoundError(f'OWL-ViT weights missing: {path}')
        self.processor = OwlViTProcessor.from_pretrained(path, local_files_only=True)
        model = OwlViTForObjectDetection.from_pretrained(path, local_files_only=True)
        self.model = model.to('cuda' if device == 'cuda' else 'cpu').eval()
        if args.precision == 'FP16' and device == 'cuda':
            self.model.half()
        self.torch = torch
        self.session = None
        if device == 'openvino':
            import onnxruntime as ort
            from PIL import Image

            tokenized = self.processor(text=[args.prompt],
                                       images=Image.new('RGB', (args.imgsz, args.imgsz)),
                                       return_tensors='pt')
            weights = path / 'model.safetensors'
            stat = weights.stat() if weights.exists() else (path / 'pytorch_model.bin').stat()
            key = json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                              args.prompt, args.imgsz])
            export = MODEL_DIR / 'exports' / ('owlvit_' +
                      hashlib.sha256(key.encode()).hexdigest()[:12] + '.onnx')
            export.parent.mkdir(parents=True, exist_ok=True)
            if not export.exists():
                class ONNXWrapper(torch.nn.Module):
                    def __init__(self, wrapped):
                        super().__init__()
                        self.wrapped = wrapped

                    def forward(self, pixel_values, input_ids, attention_mask):
                        output = self.wrapped(pixel_values=pixel_values,
                                              input_ids=input_ids,
                                              attention_mask=attention_mask)
                        return output.logits, output.pred_boxes

                print(f'[INFO] Exporting OWL-ViT to {export} for OpenVINO EP')
                torch.onnx.export(
                    ONNXWrapper(self.model),
                    (tokenized['pixel_values'], tokenized['input_ids'],
                     tokenized['attention_mask']), str(export),
                    input_names=['pixel_values', 'input_ids', 'attention_mask'],
                    output_names=['logits', 'pred_boxes'], opset_version=17,
                    dynamo=False)
            providers = resolve_providers('openvino', args.openvino_device,
                                          args.precision)
            self.session = ort.InferenceSession(str(export), providers=providers)
            if self.session.get_providers()[0] != 'OpenVINOExecutionProvider':
                raise RuntimeError('OWL-ViT ONNX session did not activate OpenVINO EP.')
            print(f'[INFO] OWL-ViT provider: {self.session.get_providers()[0]}')

    def search(self, image):
        from PIL import Image
        small, sx, sy = _max_side(image, self.args.imgsz)
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=[self.args.prompt], images=pil,
                                return_tensors='pt').to('cuda' if self.device == 'cuda' else 'cpu')
        if self.args.precision == 'FP16' and self.device == 'cuda':
            inputs['pixel_values'] = inputs['pixel_values'].half()
        if self.session is not None:
            raw = self.session.run(
                ['logits', 'pred_boxes'],
                {name: inputs[name].cpu().numpy() for name in
                 ('pixel_values', 'input_ids', 'attention_mask')})
            outputs = SimpleNamespace(logits=self.torch.from_numpy(raw[0]),
                                      pred_boxes=self.torch.from_numpy(raw[1]))
        else:
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
        sizes = self.torch.tensor([pil.size[::-1]], device=outputs.logits.device)
        result = self.processor.post_process_grounded_object_detection(
            outputs, target_sizes=sizes, threshold=self.args.confidence)[0]
        hits = []
        for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
            x1, y1, x2, y2 = box.tolist()
            hits.append(Hit(self.args.prompt[int(label)], float(score),
                            tuple(round(v) for v in (x1*sx, y1*sy, x2*sx, y2*sy))))
        return hits


class LocateSearch:
    def __init__(self, args, device):
        if args.weights:
            raise ValueError('LocateAnything uses a pinned Hugging Face snapshot, not --weights.')
        os.environ['HF_HOME'] = str(MODEL_DIR / 'huggingface')
        try:
            from libreyolo import LibreLocateAnything
        except ImportError as error:
            raise RuntimeError('Install libreyolo[vlm] for LocateAnything.') from error
        self.name = 'LocateAnything'
        self.args = args
        self.model = LibreLocateAnything(size='3b', device=device)
        self.model.set_classes(args.prompt)

    def search(self, image):
        small, sx, sy = _max_side(image, self.args.imgsz)
        result = self.model(small)
        hits = []
        if result.boxes is None:
            return hits
        for xyxy, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(float, xyxy)
            hits.append(Hit(str(result.names[int(cls)]), None,
                            tuple(round(v) for v in (x1*sx, y1*sy, x2*sx, y2*sy))))
        return hits


class OpenAISearch:
    def __init__(self, args, device):
        self.args = args
        self.provider = args.vlm_provider
        self.base_url = (args.vlm_base_url or
                         ('http://192.168.1.117:8081/v1' if self.provider == 'llamacpp'
                          else 'https://api.openai.com/v1')).rstrip('/')
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme not in ('http', 'https') or not parsed_url.netloc:
            raise ValueError('--vlm-base-url must be an http(s) URL')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'
        self.key = os.environ.get('LLAMACPP_API_KEY' if self.provider == 'llamacpp'
                                  else 'OPENAI_API_KEY')
        if self.provider == 'openai' and not self.key:
            raise RuntimeError('OPENAI_API_KEY is required for --vlm-provider openai.')
        self.model = args.vlm_model
        if self.model == 'auto':
            if self.provider == 'openai':
                self.model = 'gpt-4.1-mini'
            else:
                try:
                    response = requests.get(self.base_url + '/models',
                                            headers=self._headers(),
                                            timeout=min(args.vlm_timeout, 10))
                    response.raise_for_status()
                except requests.RequestException as error:
                    raise RuntimeError(f'Cannot read the llama.cpp model list at '
                                       f'{self.base_url}/models: {error}') from error
                models = response.json()
                entries = models.get('data') or models.get('models') or []
                if not entries:
                    raise RuntimeError('llama.cpp returned no model from /v1/models; '
                                       'pass --vlm-model explicitly.')
                self.model = entries[0].get('id') or entries[0].get('model')
                if not self.model:
                    raise RuntimeError('llama.cpp /v1/models entry has no model ID.')
        self.name = f'{self.provider}: {self.model}'
        self.last_description = ''
        self.last_images = []
        self.request_count = 0
        self.first_request_started = None
        self.ttft_s = None
        self.tps = None

    @property
    def rps(self):
        if self.first_request_started is None:
            return 0.0
        return self.request_count / max(time.perf_counter() - self.first_request_started, 1e-9)

    @staticmethod
    def _stream_reply(response, started):
        pieces = []
        ttft_s = None
        timings = {}
        for line in response.iter_lines(chunk_size=1):
            if not line or not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            chunk = json.loads(data)
            if 'error' in chunk:
                raise RuntimeError(f'VLM stream error: {str(chunk["error"])[:200]}')
            timings = chunk.get('timings') or timings
            for choice in chunk.get('choices', []):
                delta = choice.get('delta') or {}
                content = delta.get('content') or ''
                reasoning = delta.get('reasoning_content') or ''
                if ttft_s is None and (content or reasoning):
                    ttft_s = time.perf_counter() - started
                if content:
                    pieces.append(content)
        return ''.join(pieces), ttft_s, timings

    def status_line(self):
        ttft = f'{self.ttft_s:.2f}s' if self.ttft_s is not None else 'N/A'
        tps = f'{self.tps:.1f} tok/s' if self.tps is not None else 'N/A'
        return (f'VLM TTFT: {ttft} | TPS: {tps} | '
                f'RPS: {self.rps:.2f} | Requests: {self.request_count}')

    def _headers(self):
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = f'Bearer {self.key}'
        return headers

    @staticmethod
    def _verdict(text):
        text = text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        answer = re.match(r'^(?:(?:final\s+)?answer\s*:\s*)?["\']?(YES|NO)\b',
                          text, re.IGNORECASE)
        if answer:
            present = answer.group(1).upper() == 'YES'
            return {'present': present, 'description': 'YES' if present else 'NO'}
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find('{'), text.rfind('}')
            if start < 0 or end < start:
                raise RuntimeError(f'VLM did not return YES or NO: {text[:200]}')
            try:
                result = json.loads(text[start:end+1])
            except json.JSONDecodeError as error:
                raise RuntimeError(f'VLM did not return YES or NO: {text[:200]}') from error
        if isinstance(result, dict) and type(result.get('present')) is bool:
            return {'present': result['present'],
                    'description': str(result.get('description') or
                                       ('YES' if result['present'] else 'NO'))}
        raise RuntimeError(f'VLM did not return YES or NO: {text[:200]}')

    def _search_prompt(self, images, prompt):
        if isinstance(images, np.ndarray):
            images = [images]
        self.last_images = []
        instruction = (f'Is this visibly present in any supplied image: '
                       f'{prompt}? Answer exactly YES or NO. '
                       'No explanation, punctuation, or JSON.')
        content = [{'type': 'text', 'text': instruction}]
        for index, image in enumerate(images):
            small, _, _ = _max_side(image, self.args.imgsz)
            ok, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError('Could not encode a VLM frame.')
            self.last_images.append(small.copy())
            url = 'data:image/jpeg;base64,' + base64.b64encode(jpg.tobytes()).decode('ascii')
            content.append({'type': 'text', 'text': f'Frame {index + 1}'})
            content.append({'type': 'image_url', 'image_url': {'url': url}})
        if self.provider == 'llamacpp':
            url = self.base_url + '/chat/completions'
            body = {'model': self.model,
                    'messages': [{'role': 'user', 'content': content}],
                    'temperature': 0, 'max_tokens': 64, 'stream': True,
                    'stream_options': {'include_usage': True},
                    'reasoning_effort': 'none',
                    'chat_template_kwargs': {'enable_thinking': False}}
        else:
            url = self.base_url + '/responses'
            openai_content = []
            for item in content:
                if item['type'] == 'text':
                    openai_content.append({'type': 'input_text', 'text': item['text']})
                else:
                    openai_content.append({'type': 'input_image',
                                           'image_url': item['image_url']['url']})
            body = {'model': self.model, 'store': False,
                    'input': [{'role': 'user', 'content': openai_content}],
                    'max_output_tokens': 16}
        started = time.perf_counter()
        if self.first_request_started is None:
            self.first_request_started = started
        try:
            response = requests.post(url, headers=self._headers(), json=body,
                                     timeout=self.args.vlm_timeout,
                                     stream=self.provider == 'llamacpp')
            response.raise_for_status()
            if self.provider == 'llamacpp':
                try:
                    text, self.ttft_s, timings = self._stream_reply(response, started)
                finally:
                    response.close()
                predicted_n = timings.get('predicted_n')
                predicted_ms = timings.get('predicted_ms')
                self.tps = (1000 * predicted_n / predicted_ms
                            if isinstance(predicted_n, (int, float)) and
                            isinstance(predicted_ms, (int, float)) and predicted_ms > 0
                            else None)
            else:
                payload = response.json()
                texts = [piece.get('text', '') for item in payload.get('output', [])
                         for piece in item.get('content', [])
                         if piece.get('type') == 'output_text']
                text = ''.join(texts)
                self.ttft_s = None
                self.tps = None
        except requests.Timeout as error:
            raise RuntimeError(f'VLM request timed out after {self.args.vlm_timeout:g}s '
                               f'at {url}. Increase --vlm-timeout or reduce '
                               '--imgsz/--clip-frames.') from error
        except requests.ConnectionError as error:
            raise RuntimeError(f'VLM server disconnected or is unavailable at {url}. '
                               'Check its health and model memory usage.') from error
        except requests.HTTPError as error:
            raise RuntimeError(f'VLM server returned HTTP {response.status_code} '
                               f'at {url}: {response.text[:300]}') from error
        if not text:
            raise RuntimeError('VLM returned no answer. If llama.cpp used its output '
                               'for reasoning, start it with --reasoning off.')
        verdict = self._verdict(text)
        self.last_description = verdict['description']
        self.request_count += 1
        return Hit(prompt, description=verdict['description']) if verdict['present'] else None

    def search(self, images):
        prompts = (['; '.join(self.args.prompt)]
                   if self.args.vlm_prompt_mode == 'combine' else self.args.prompt)
        hits = []
        results = []
        for prompt in prompts:
            hit = self._search_prompt(images, prompt)
            results.append((prompt, bool(hit)))
            if hit is not None:
                hits.append(hit)
        self.last_prompt_results = results
        self.last_description = ' | '.join(
            f'{prompt}: {"YES" if present else "NO"}' for prompt, present in results)
        return hits


BACKENDS = {'owlvit': OwlSearch, 'yoloe': YoloSearch,
            'yolo-world': YoloSearch, 'locateanything': LocateSearch,
            'openai': OpenAISearch}


class PersonFilter:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = _load_yolo(args.prefilter_weights, 'person', None, device, args)
        names = self.model.names
        self.person_ids = [int(index) for index, name in names.items()
                           if str(name).casefold() == 'person']
        if not self.person_ids:
            raise ValueError('The prefilter model has no "person" class.')
        self.visuals = []

    def regions(self, frame):
        result = self.model.predict(frame, classes=self.person_ids,
                                    device='cpu' if self.device == 'openvino'
                                    else '0' if self.device == 'cuda' else 'cpu',
                                    imgsz=self.args.imgsz,
                                    conf=self.args.prefilter_confidence,
                                    half=self.device == 'cuda' and self.args.precision == 'FP16',
                                    verbose=False)[0]
        height, width = frame.shape[:2]
        regions = []
        self.visuals = []
        for hit in _boxes_from_ultralytics(result, result.names):
            x1, y1, x2, y2 = hit.box
            dx = round((x2 - x1) * self.args.crop_expand)
            dy = round((y2 - y1) * self.args.crop_expand)
            region = (max(0, x1-dx), max(0, y1-dy),
                      min(width, x2+dx), min(height, y2+dy))
            if region[2] > region[0] and region[3] > region[1]:
                regions.append((frame[region[1]:region[3], region[0]:region[2]], region))
                self.visuals.append((hit.box, region))
        return regions


def _fitted_line(canvas, text, x, y, width, scale=0.52, color=(235, 240, 245)):
    full = text
    while text and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
        text = text[:-1]
    if text != full and len(text) > 3:
        text = text[:-3] + '...'
    if text:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)


def _wrap_text(text, width, scale=0.51):
    lines = []
    current = ''
    for word in text.split():
        candidate = f'{current} {word}'.strip()
        if current and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX,
                                       scale, 1)[0][0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def compose_canvas(frame, hits, args, backend_name, device, frame_index,
                   infer_fps, streak, source_fps, description='', prefilter_visuals=(),
                   vlm_stats=None, vlm_preview=None):
    height, width = frame.shape[:2]
    canvas = np.full((height + STRIP_HEIGHT, width + PANEL_WIDTH, 3),
                     (20, 25, 34), dtype=np.uint8)
    canvas[:height, :width] = frame
    cv2.line(canvas, (width, 0), (width, height), (95, 105, 120), 1)
    cv2.line(canvas, (0, height), (width + PANEL_WIDTH, height),
             (95, 105, 120), 1)
    if args.show_prefilter:
        for person_box, crop_box in prefilter_visuals:
            px1, py1, px2, py2 = person_box
            cx1, cy1, cx2, cy2 = crop_box
            cv2.rectangle(canvas, (cx1, cy1), (cx2-1, cy2-1), (255, 220, 0), 2)
            cv2.rectangle(canvas, (px1, py1), (px2-1, py2-1), (0, 165, 255), 2)
            _fitted_line(canvas, 'person', max(0, px1+4), max(17, py1+18),
                         max(20, width-px1-8), color=(0, 165, 255))
    for hit in hits:
        if hit.box is None:
            continue
        x1, y1, x2, y2 = hit.box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 210, 80), 2)
        label = f'{hit.label} {hit.score:.2f}' if hit.score is not None else hit.label
        label_y = y1 - 6 if y1 >= 25 else y1 + 20
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        cv2.rectangle(canvas, (x1, label_y-17),
                      (min(width-1, x1+text_width+8), label_y+4),
                      (25, 85, 40), -1)
        _fitted_line(canvas, label, x1+4, label_y, max(20, width-x1-12),
                     color=(255, 255, 255))
    if args.alert and streak >= args.alert_after:
        thickness = max(8, min(width, height) // 45)
        cv2.rectangle(canvas, (thickness//2, thickness//2),
                      (width-1-thickness//2, height-1-thickness//2),
                      (0, 0, 255), thickness)
    panel_x = width + 12
    _fitted_line(canvas, f'Targets: {len(hits)}', panel_x, 28, PANEL_WIDTH - 24,
                 scale=0.64)
    y = 52
    if args.backend == 'openai' and vlm_preview is not None and vlm_preview.size:
        _fitted_line(canvas, 'VLM input (latest frame)', panel_x, y+16,
                     PANEL_WIDTH-24, color=(150, 205, 255))
        y += 22
        preview_h = min(max(1, int(height * 0.34)), 220)
        ratio = min((PANEL_WIDTH-24)/vlm_preview.shape[1],
                    preview_h/vlm_preview.shape[0])
        preview_size = (max(1, round(vlm_preview.shape[1]*ratio)),
                        max(1, round(vlm_preview.shape[0]*ratio)))
        preview = cv2.resize(vlm_preview, preview_size, interpolation=cv2.INTER_AREA)
        canvas[y:y+preview_size[1], panel_x:panel_x+preview_size[0]] = preview
        y += preview_size[1] + 10
    if not hits:
        _fitted_line(canvas, 'No target in last inference', panel_x, y+18,
                     PANEL_WIDTH-24)
        y += 32
        for line in _wrap_text(description, PANEL_WIDTH-24)[:8]:
            if y + 18 >= height:
                break
            _fitted_line(canvas, line, panel_x, y+16, PANEL_WIDTH-24)
            y += 21
        if args.show_prefilter and prefilter_visuals and y + 70 < height:
            _, (x1, y1, x2, y2) = prefilter_visuals[0]
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                _fitted_line(canvas, f'Prefilter crop (of {len(prefilter_visuals)})',
                             panel_x, y+18, PANEL_WIDTH-24, color=(255, 220, 0))
                y += 27
                ratio = min((PANEL_WIDTH-24)/crop.shape[1],
                            max(1, height-y-12)/crop.shape[0], 1.0)
                size = (max(1, round(crop.shape[1]*ratio)),
                        max(1, round(crop.shape[0]*ratio)))
                canvas[y:y+size[1], panel_x:panel_x+size[0]] = cv2.resize(
                    crop, size, interpolation=cv2.INTER_AREA)
    for hit in hits[:3]:
        if y >= height - 40:
            break
        title = f'{hit.label}  {hit.score:.2f}' if hit.score is not None else hit.label
        _fitted_line(canvas, title, panel_x, y+18, PANEL_WIDTH-24,
                     color=(100, 235, 150))
        y += 26
        if hit.box is not None:
            x1, y1, x2, y2 = hit.box
            crop = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
            if crop.size:
                target_w = PANEL_WIDTH - 24
                target_h = min(max(0, height - y - 22), 130)
                if target_h > 4:
                    ratio = min(target_w / crop.shape[1], target_h / crop.shape[0])
                    size = (max(1, round(crop.shape[1]*ratio)),
                            max(1, round(crop.shape[0]*ratio)))
                    preview = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
                    canvas[y:y+size[1], panel_x:panel_x+size[0]] = preview
                    y += size[1] + 12
        else:
            for line in _wrap_text(hit.description, PANEL_WIDTH-24)[:5]:
                if y + 18 >= height:
                    break
                _fitted_line(canvas, line, panel_x, y+16, PANEL_WIDTH-24)
                y += 21
            y += 10
    mode = f'{args.backend}/{args.vlm_mode}' if args.backend == 'openai' else args.backend
    endpoint = 'llama.cpp LAN' if args.vlm_provider == 'llamacpp' else 'OpenAI remote'
    device_label = (f'{endpoint}; prefilter: {device} {args.precision}'
                    if args.backend == 'openai' and args.prefilter == 'person'
                    else endpoint if args.backend == 'openai'
                    else f'{device} {args.precision}')
    _fitted_line(canvas, f'Model: {backend_name} | Mode: {mode} | Device: {device_label}', 10, height+24,
                 width + PANEL_WIDTH - 20)
    _fitted_line(canvas, f'Prompt: {"; ".join(args.prompt)}', 10, height+47,
                 width + PANEL_WIDTH - 20)
    status_y = height + 69
    if vlm_stats is not None:
        _fitted_line(canvas, vlm_stats.status_line(), 10, height+69,
                     width + PANEL_WIDTH - 20, scale=0.46)
        status_y = height + 88
    _fitted_line(canvas, f'Frame: {frame_index} | Video: {source_fps:.1f} FPS | '
                 f'Infer: {infer_fps:.1f} FPS | Every {args.infer_frame_interval} frame | '
                 f'Filter: {args.prefilter}{" (shown)" if args.show_prefilter else ""} | '
                 f'Streak: {streak}/{args.alert_after}',
                 10, status_y, width + PANEL_WIDTH - 20, scale=0.46)
    return canvas


def _output_path(args, source):
    if args.output:
        path = Path(args.output)
    elif isinstance(source, str) and Path(source).is_file():
        path = Path(args.output_dir) / f'{Path(source).stem}_search_{args.backend}.mp4'
    else:
        name = (f'camera_{source}' if isinstance(source, int) else
                urlparse(str(source)).hostname or 'stream')
        path = Path(args.output_dir) / f'{name}_search_{args.backend}_{datetime.now():%Y%m%d_%H%M%S}.mp4'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_search(searcher, frame, regions, args, clip_buffer):
    hits = []
    if not regions and args.backend == 'openai':
        searcher.last_description = 'No person crop passed the filter.'
    if args.backend == 'openai' and args.vlm_mode == 'clip':
        if regions and clip_buffer:
            hits = searcher.search(list(clip_buffer))
    else:
        for crop, region in regions:
            for hit in searcher.search(crop):
                if hit.box is not None:
                    x1, y1, x2, y2 = hit.box
                    hit.box = (x1+region[0], y1+region[1],
                               x2+region[0], y2+region[1])
                hits.append(hit)
    return hits


def main(argv=None):
    args = parse_args(argv)
    if args.list_devices:
        print(f'[INFO] Found devices: {get_available_devices(max_index=10)}')
        return
    device = resolve_device(args)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prefilter = PersonFilter(args, device) if args.prefilter == 'person' else None
    searcher = BACKENDS[args.backend](args, device)
    source = parse_video_device(args.input_device)
    if args.output and isinstance(source, str) and Path(source).is_file():
        if Path(args.output).resolve() == Path(source).resolve():
            raise ValueError('Output video must not overwrite the input video.')
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video input: {source}')
    writer = None
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else 25.0
        frame_index = 0
        streak = 0
        hits = []
        infer_fps = 0.0
        clip_buffer = deque(maxlen=args.clip_frames)
        input_is_file = isinstance(source, str) and Path(source).is_file()
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
            height, width = frame.shape[:2]
            clip_mode = args.backend == 'openai' and args.vlm_mode == 'clip'
            inference_frame = frame_index % args.infer_frame_interval == 0
            start = time.perf_counter() if inference_frame else None
            prefilter_visuals = []
            if clip_mode or inference_frame:
                regions = (prefilter.regions(frame) if prefilter is not None else
                           [(frame, (0, 0, width, height))])
                if args.show_prefilter and prefilter is not None:
                    prefilter_visuals = prefilter.visuals
                if clip_mode:
                    if regions:
                        # A clip follows the largest current person crop when
                        # filtering is enabled; otherwise it follows the frame.
                        largest = max(regions, key=lambda item: item[0].shape[0] * item[0].shape[1])
                        clip_buffer.append(largest[0].copy())
                    else:
                        clip_buffer.clear()
            if inference_frame:
                hits = run_search(searcher, frame, regions, args, clip_buffer)
                infer_fps = 1 / max(time.perf_counter()-start, 1e-9)
                streak = streak + 1 if hits else 0
                if hits:
                    print(f'[HIT] frame={frame_index} targets={len(hits)} streak={streak}')
            canvas = compose_canvas(frame, hits, args, searcher.name, device,
                                    frame_index, infer_fps, streak, source_fps,
                                    getattr(searcher, 'last_description', ''),
                                    prefilter_visuals,
                                    searcher if args.backend == 'openai' else None,
                                    (searcher.last_images[-1] if args.backend == 'openai'
                                     and searcher.last_images else None))
            if writer is None and not args.no_output:
                path = _output_path(args, source)
                writer = EncodedVideoWriter(path,
                                            (width+PANEL_WIDTH, height+STRIP_HEIGHT),
                                            source_fps, args.video_codec,
                                            args.video_crf, args.video_preset)
                print(f'[INFO] OUTPUT: {path}')
            if writer is not None:
                writer.write(canvas)
            if not args.no_display:
                cv2.imshow(f'Video search: {source}', canvas)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
