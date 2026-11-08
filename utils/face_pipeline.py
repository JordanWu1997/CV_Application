"""Shared face results, optional tracking, and video layout helpers.

Model pipelines return FaceFrame objects. Renderers consume those objects, so
person detection, ReID, and pose components can use the same records directly.
"""

from dataclasses import dataclass, field, replace
from math import ceil, sqrt
from typing import Any

import cv2
import numpy as np


@dataclass
class FaceRecord:
    bbox: tuple
    landmarks: Any
    confidence: float
    crop: Any = None
    track_id: int | None = None
    attributes: dict = field(default_factory=dict)
    reference_crop: Any = None
    mask: Any = None
    extras: dict = field(default_factory=dict)


@dataclass
class FaceFrame:
    frame_index: int
    faces: list[FaceRecord] = field(default_factory=list)
    inferred: bool = False
    inference_seconds: float = 0.0
    tracking: bool = False
    extras: dict = field(default_factory=dict)


def crop_face(frame, bbox, padding=0.12):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return overlap / max(1e-6, area_a + area_b - overlap)


class FaceTracker:
    """Greedy IoU IDs at detection frames; optical flow between detections."""

    def __init__(self, iou_threshold=0.25):
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.records = []
        self.previous_gray = None

    def update(self, frame, detections=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if detections is not None:
            candidates = sorted(
                ((_iou(old.bbox, new.bbox), old.track_id, new_idx)
                 for old in self.records for new_idx, new in enumerate(detections)),
                reverse=True)
            used_ids, used_new = set(), set()
            for score, track_id, new_idx in candidates:
                if score < self.iou_threshold:
                    break
                if track_id not in used_ids and new_idx not in used_new:
                    detections[new_idx].track_id = track_id
                    used_ids.add(track_id)
                    used_new.add(new_idx)
            for item in detections:
                if item.track_id is None:
                    item.track_id = self.next_id
                    self.next_id += 1
            self.records = detections
        elif self.previous_gray is not None:
            advanced = []
            for item in self.records:
                points = np.asarray(item.landmarks, dtype=np.float32).reshape(-1, 1, 2)
                if len(points) == 0:
                    advanced.append(item)
                    continue
                moved, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.previous_gray, gray, points, None)
                if moved is None or status is None or not np.any(status):
                    advanced.append(item)
                    continue
                good = status.reshape(-1).astype(bool)
                delta = np.median((moved - points).reshape(-1, 2)[good], axis=0)
                if not np.all(np.isfinite(delta)):
                    advanced.append(item)
                    continue
                dx, dy = float(delta[0]), float(delta[1])
                bbox = (item.bbox[0] + dx, item.bbox[1] + dy,
                        item.bbox[2] + dx, item.bbox[3] + dy)
                landmarks = np.asarray(item.landmarks, dtype=np.float32) + delta
                advanced.append(replace(item, bbox=bbox, landmarks=landmarks,
                                        crop=crop_face(frame, bbox)))
            self.records = advanced
        self.previous_gray = gray
        return self.records


def output_layout(frame_shape, target_groups=None, history_height=0):
    h, w = frame_shape[:2]
    side_width = max(360, w // 2)
    full_width = w + side_width
    gallery_height = 0
    if target_groups:
        group_width = full_width // len(target_groups)
        thumb_size = max(8, min(68, group_width - 20))
        pitch = thumb_size + 18
        cols = max(1, (group_width - 16) // pitch)
        rows = max(ceil(len(group['faces']) / cols) for group in target_groups)
        gallery_height = 48 + rows * (thumb_size + 36)
    return full_width, h + gallery_height + history_height, side_width, gallery_height


def _text(image, value, x, y, scale=0.65, color=(240, 245, 250), max_width=None):
    value = str(value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if max_width is not None:
        while value and cv2.getTextSize(value, font, scale, 1)[0][0] > max_width:
            value = value[:-4] + '...' if len(value) > 4 else value[:-1]
    cv2.putText(image, value, (int(x), int(y)), font, scale, color, 2, cv2.LINE_AA)


def _tile(canvas, image, x, y, size, title):
    if image is not None and image.size:
        thumb = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        if thumb.ndim == 2:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        elif thumb.shape[2] == 4:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGRA2BGR)
        canvas[y:y + size, x:x + size] = thumb
    else:
        cv2.rectangle(canvas, (x, y), (x + size, y + size), (55, 60, 70), -1)
    _text(canvas, title, x, y + size + 18, 0.5, max_width=size)


def render_face_frame(frame, result, mode, layout, target_groups=None,
                      infer_interval=1, history=None):
    """Render without placing cards or galleries over the source image."""
    h, w = frame.shape[:2]
    out_w, out_h, side_width, gallery_height = layout
    canvas = np.full((out_h, out_w, 3), (20, 25, 34), dtype=np.uint8)
    canvas[:h, :w] = frame
    fps = 1 / result.inference_seconds if result.inference_seconds > 0 else 0.0
    tracking = result.tracking
    status = (f'{mode}  {len(result.faces)} faces  {fps:.1f} infer FPS  '
              f'track {"ON" if tracking else "OFF"}')
    _text(canvas, status, w + 10, 31, 0.61, max_width=side_width - 20)
    for index, face in enumerate(result.faces, 1):
        color = ((75, 220, 110) if face.attributes.get('matched') else (70, 100, 230)) if mode == 'FS' else (75, 220, 220)
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        face_id = face.track_id if face.track_id is not None else index
        tag = f'#{face_id}'
        cv2.rectangle(canvas, (max(0, x1), max(0, y1 - 28)),
                      (max(0, x1) + 54, max(0, y1)), color, -1)
        _text(canvas, tag, max(0, x1) + 4, max(18, y1 - 6), 0.6)

    # Adaptive grid in the side rail: every face gets a disjoint cell.
    count = len(result.faces)
    if count:
        rail_top = 48
        rail_height = max(1, h - rail_top)
        cols = max(1, ceil(sqrt(count * side_width / rail_height)))
        rows = ceil(count / cols)
        cell_w, cell_h = side_width // cols, rail_height // rows
        for index, face in enumerate(result.faces):
            col, row = index % cols, index // cols
            x, y = w + col * cell_w, rail_top + row * cell_h
            if cell_w < 8 or cell_h < 8:
                continue
            cv2.rectangle(canvas, (x + 3, y + 3),
                          (x + cell_w - 4, y + cell_h - 4), (48, 57, 69), -1)
            face_id = face.track_id if face.track_id is not None else index + 1
            title = f'FACE #{face_id}  det {face.confidence:.2f}'
            if cell_h >= 34:
                _text(canvas, title, x + 10, y + 27, 0.62, max_width=cell_w - 20)
            tile_size = min(112, max(0, cell_w // (3 if mode == 'FD' else 2) - 14),
                            max(0, cell_h - 150))
            if tile_size >= 24:
                _tile(canvas, face.crop, x + 10, y + 37, tile_size, 'INPUT')
                if mode == 'FS':
                    _tile(canvas, face.reference_crop, x + 20 + tile_size,
                          y + 37, tile_size, 'MATCH')
                elif face.mask is not None:
                    _tile(canvas, face.mask, x + 20 + tile_size,
                          y + 37, tile_size, 'PARSING')
            if mode == 'FS':
                label = (f"{face.attributes.get('group', 'NO MATCH')}  "
                         f"sim {face.attributes.get('similarity', -1):.2f}")
                detail = face.attributes.get('reference_label', '')
            else:
                label = (f"{face.attributes.get('gender', '?')}  "
                         f"age {face.attributes.get('age', '?')}  "
                         f"{face.attributes.get('emotion', '?')}")
                detail = (f"gaze {face.attributes.get('pitch', '?')}, "
                          f"{face.attributes.get('yaw', '?')}")
            if cell_h >= 95:
                label_y = min(y + cell_h - 47,
                              y + max(95, 37 + tile_size + 42))
                _text(canvas, label, x + 10, label_y, 0.62,
                      max_width=cell_w - 20)
                _text(canvas, detail, x + 10,
                      min(y + cell_h - 20, label_y + 27), 0.55,
                      max_width=cell_w - 20)
            elif cell_h >= 62:
                _text(canvas, label, x + 10, y + cell_h - 14, 0.52,
                      max_width=cell_w - 20)
    else:
        _text(canvas, 'No faces detected', w + 16, 83, 0.8)

    if target_groups:
        top = h
        group_width = out_w // len(target_groups)
        thumb_size = max(8, min(68, group_width - 20))
        pitch = thumb_size + 18
        row_height = thumb_size + 36
        for group_idx, group in enumerate(target_groups):
            gx = group_idx * group_width
            _text(canvas, f"{group['name']} ({len(group['faces'])} references)",
                  gx + 10, top + 29, 0.68, max_width=group_width - 20)
            cols = max(1, (group_width - 16) // pitch)
            for ref_idx, image in enumerate(group['faces']):
                x = gx + 10 + (ref_idx % cols) * pitch
                y = top + 43 + (ref_idx // cols) * row_height
                _tile(canvas, image, x, y, thumb_size, str(ref_idx + 1))
    if history:
        top = h + gallery_height
        _text(canvas, 'RECENT FACES', 10, top + 27, 0.65)
        for index, (image, frame_index) in enumerate(history):
            x = 170 + index * 86
            if x + 68 > out_w:
                break
            _tile(canvas, image, x, top + 8, 68, str(frame_index))
    return canvas
