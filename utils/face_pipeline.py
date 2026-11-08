"""Shared face results, optional tracking, and video layout helpers.

Model pipelines return FaceFrame objects. Renderers consume those objects, so
person detection, ReID, and pose components can use the same records directly.
"""

from dataclasses import dataclass, field, replace
from collections import Counter, deque
from math import ceil, degrees, isfinite, sin, sqrt
from typing import Any

import cv2
import numpy as np


BOX_STATUS_HEIGHT = 42


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


class TrackAttributeVoter:
    """Smooth FD estimates over recent detections of the same track."""

    def __init__(self, window=5):
        self.window = window
        self.history = {}

    def update(self, records):
        active = {item.track_id for item in records}
        self.history = {key: value for key, value in self.history.items()
                        if key in active}
        for item in records:
            samples = self.history.setdefault(item.track_id, deque(maxlen=self.window))
            samples.append(item.attributes.copy())
            for key in ('gender', 'emotion'):
                values = [sample.get(key) for sample in samples
                          if sample.get(key) not in (None, 'N/A')]
                if values:
                    item.attributes[key] = Counter(values).most_common(1)[0][0]
            for key in ('age', 'pitch', 'yaw', 'head_pitch', 'head_yaw', 'head_roll'):
                values = [sample.get(key) for sample in samples]
                values = [value for value in values
                          if isinstance(value, (int, float)) and isfinite(value)]
                if values:
                    item.attributes[key] = float(np.median(values))
            if 'face_states' in item.attributes:
                states = [sample['face_states'] for sample in samples
                          if sample.get('face_states')]
                item.attributes['face_states'] = {
                    key: float(np.mean([state[key] for state in states]))
                    for key in states[-1]
                }
        return records


class TrackIdentityVoter:
    """Require repeated FS matches before assigning a group to a track."""

    def __init__(self, window=5, min_votes=3):
        self.window = window
        self.min_votes = min_votes
        self.history = {}

    def update(self, records, target_groups):
        active = {item.track_id for item in records}
        self.history = {key: value for key, value in self.history.items()
                        if key in active}
        for item in records:
            samples = self.history.setdefault(item.track_id, deque(maxlen=self.window))
            samples.append((item.extras['group_index'],
                            item.attributes['reference_index'],
                            item.attributes['similarity']))
            votes = Counter(group for group, _, _ in samples if group is not None)
            winner, count = votes.most_common(1)[0] if votes else (None, 0)
            item.attributes['vote_count'] = count
            item.attributes['vote_total'] = len(samples)
            if winner is None or count < self.min_votes or count <= len(samples) // 2:
                item.attributes.update(matched=False, group='NO MATCH',
                                       reference_index=None, reference_label='')
                item.reference_crop = None
                continue
            winning = [(reference, similarity) for group, reference, similarity in samples
                       if group == winner]
            reference = max(winning, key=lambda value: value[1])[0]
            group = target_groups[winner]
            item.attributes.update(matched=True, group=group['name'],
                                   reference_index=reference,
                                   reference_label=group['labels'][reference],
                                   similarity=float(np.mean([value[1] for value in winning])))
            item.reference_crop = group['faces'][reference]
        return records


def output_layout(frame_shape, target_groups=None, history_height=0,
                  min_side_width=360, result_layout='panel'):
    h, w = frame_shape[:2]
    if result_layout == 'box':
        if not target_groups:
            return w, h + BOX_STATUS_HEIGHT, 0, 0
        group_width = w // len(target_groups)
        thumb_size = max(8, min(68, group_width - 20))
        pitch = thumb_size + 18
        cols = max(1, (group_width - 16) // pitch)
        rows = max(ceil(len(group['faces']) / cols) for group in target_groups)
        gallery_height = 48 + rows * (thumb_size + 36)
        return w, h + BOX_STATUS_HEIGHT + gallery_height, 0, gallery_height
    side_width = max(min_side_width, w // 2)
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


def _text(image, value, x, y, scale=0.65, color=(240, 245, 250),
          max_width=None, thickness=2):
    value = str(value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if max_width is not None:
        while value and cv2.getTextSize(value, font, scale, 1)[0][0] > max_width:
            value = value[:-4] + '...' if len(value) > 4 else value[:-1]
    cv2.putText(image, value, (int(x), int(y)), font, scale, color,
                thickness, cv2.LINE_AA)


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
    if title:
        _text(canvas, title, x, y + size + 18, 0.5, max_width=size)


def _angle_text(value):
    return f'{degrees(value):+.0f}' if value is not None and isfinite(value) else '?'


def _fd_card_lines(face, face_id):
    attrs = face.attributes
    lines = [f'FACE #{face_id}  det {face.confidence:.2f}']
    details = []
    if attrs.get('gender') is not None:
        details.append(str(attrs['gender']))
    if attrs.get('age') is not None:
        details.append(f"age {attrs['age']}")
    if attrs.get('emotion') not in (None, 'N/A'):
        details.append(str(attrs['emotion']))
    if details:
        lines.append('  '.join(details))
    if attrs.get('pitch') is not None and attrs.get('yaw') is not None:
        lines.append(f"gaze P{_angle_text(attrs['pitch'])} Y{_angle_text(attrs['yaw'])}")
    if all(attrs.get(key) is not None for key in ('head_pitch', 'head_yaw', 'head_roll')):
        lines.append('head P{:+.0f} Y{:+.0f} R{:+.0f}'.format(
            attrs['head_pitch'], attrs['head_yaw'], attrs['head_roll']))
    if attrs.get('quality_score') is not None:
        lines.append(f"quality {attrs['quality_score']:.3f}")
    states = attrs.get('face_states')
    if states:
        lines.extend((
            f"eyes L{states['left_eye_open']:.2f} R{states['right_eye_open']:.2f}",
            f"glasses {states['eyeglasses']:.2f} sun {states['sunglasses']:.2f}",
            f"mask {states['mask']:.2f}",
        ))
    return lines


def _fd_has_results(face):
    attrs = face.attributes
    return (face.mask is not None
            or attrs.get('age') is not None or attrs.get('gender') is not None
            or attrs.get('emotion') not in (None, 'N/A')
            or attrs.get('pitch') is not None
            or attrs.get('head_pitch') is not None
            or attrs.get('quality_score') is not None
            or bool(attrs.get('face_states')))


def _fs_card_lines(face, face_id):
    attrs = face.attributes
    lines = [f'FACE #{face_id}  det {face.confidence:.2f}',
             f"{attrs.get('group', 'NO MATCH')}  sim {attrs.get('similarity', -1):.2f}"]
    if attrs.get('reference_label'):
        lines.append(f"ref {attrs['reference_label']}")
    if 'vote_total' in attrs:
        lines.append(f"votes {attrs['vote_count']}/{attrs['vote_total']}")
    if attrs.get('pitch') is not None and attrs.get('yaw') is not None:
        lines.append(f"gaze P{_angle_text(attrs['pitch'])} Y{_angle_text(attrs['yaw'])}")
    if all(attrs.get(key) is not None for key in ('head_pitch', 'head_yaw', 'head_roll')):
        lines.append('head P{:+.0f} Y{:+.0f} R{:+.0f}'.format(
            attrs['head_pitch'], attrs['head_yaw'], attrs['head_roll']))
    return lines


def _brief_card_lines(face, face_id, mode):
    attrs = face.attributes
    if mode == 'FS':
        return [f'FACE #{face_id}',
                f"{attrs.get('group', 'NO MATCH')} {attrs.get('similarity', -1):.2f}"]
    return [f'FACE #{face_id}',
            f"{attrs.get('gender', '?')} age {attrs.get('age', '?')} {attrs.get('emotion', '?')}"]


def head_pose_box_points(face):
    """Project a rotated 3D face box using UniFace's pitch/yaw/roll convention."""
    attrs = face.attributes
    angles = [attrs.get(key) for key in ('head_pitch', 'head_yaw', 'head_roll')]
    if any(value is None or not isfinite(value) for value in angles):
        return None
    pitch, yaw, roll = np.radians(angles)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    # Image Y points down: negate pitch so positive UniFace pitch looks down.
    rotate_x = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    rotate_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotate_z = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    x1, y1, x2, y2 = face.bbox
    size = max(1.0, min(x2 - x1, y2 - y1))
    corners = np.array([(x, y, z) for z in (0.32, -0.32)
                        for x, y in ((-0.43, -0.5), (0.43, -0.5),
                                     (0.43, 0.5), (-0.43, 0.5))]) * size
    rotated = corners @ (rotate_z @ rotate_y @ rotate_x).T
    distance = 3.5 * size
    scale = distance / (distance - rotated[:, 2])
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    return np.rint(rotated[:, :2] * scale[:, None] + center).astype(np.int32)


def draw_head_pose_box(image, face):
    points = head_pose_box_points(face)
    if points is None:
        return
    cv2.polylines(image, [points[4:8]], True, (120, 150, 170), 1, cv2.LINE_AA)
    for index in range(4):
        cv2.line(image, tuple(points[index]), tuple(points[index + 4]),
                 (80, 190, 255), 2, cv2.LINE_AA)
    cv2.polylines(image, [points[:4]], True, (40, 230, 255), 2, cv2.LINE_AA)


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _face_color(face, mode):
    if mode == 'FS':
        return (75, 220, 110) if face.attributes.get('matched') else (70, 100, 230)
    return (75, 220, 220)


def plan_fd_card_positions(faces, card_sizes, canvas_size, source_width):
    """Place cards right, bottom, left, top; reserve boxes and earlier cards."""
    out_w, out_h = canvas_size
    obstacles = [(source_width, 0, out_w, 46)]
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        pad = max(4, 0.2 * max(x2 - x1, y2 - y1)) if head_pose_box_points(face) is not None else 4
        obstacles.append((x1 - pad, y1 - max(32, pad), x2 + pad, y2 + pad))
    positions = []
    for face, (card_w, card_h) in zip(faces, card_sizes):
        x1, y1, x2, y2 = [int(value) for value in face.bbox]
        chosen = None
        # Slide along each side before moving clockwise to the next side.
        for direction in ('right', 'bottom', 'left', 'top'):
            if direction in ('right', 'left'):
                x = x2 + 8 if direction == 'right' else x1 - card_w - 8
                anchors = (y1, y2 - card_h, y1 - card_h // 2,
                           y2, y1 - card_h)
                candidates = ((x, max(0, min(out_h - card_h, y))) for y in anchors)
            else:
                y = y2 + 8 if direction == 'bottom' else y1 - card_h - 36
                anchors = (x1, x2 - card_w, x1 - card_w // 2,
                           x2, x1 - card_w)
                candidates = ((max(0, min(out_w - card_w, x)), y) for x in anchors)
            for x, y in candidates:
                rect = (x, y, x + card_w, y + card_h)
                if (x >= 0 and y >= 0 and rect[2] <= out_w and rect[3] <= out_h
                        and not any(_overlaps(rect, other) for other in obstacles)):
                    chosen = rect
                    break
            if chosen is not None:
                break
        if chosen is None:
            # The attached margin provides a last place for crowded frames.
            for y in range(50, max(50, out_h - card_h + 1), 8):
                rect = (source_width + 8, y, source_width + 8 + card_w,
                        y + card_h)
                if (rect[2] <= out_w and rect[3] <= out_h
                        and not any(_overlaps(rect, other) for other in obstacles)):
                    chosen = rect
                    break
        positions.append(chosen)
        if chosen is not None:
            obstacles.append(chosen)
    return positions


def _draw_result_card(canvas, face, lines, rect, mode, show_thumbnails=True,
                      image_first=False, inline_reference=False,
                      inline_parsing=False):
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (29, 37, 47), -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), _face_color(face, mode), 2)
    second = face.mask if mode == 'FD' else face.reference_crop
    line_step = min(16 if image_first else 17,
                    max(9, (y2 - y1 - 20) // max(1, len(lines))))
    text_top = y1 + 18
    text_width = x2 - x1 - 14
    if inline_reference and face.reference_crop is not None:
        thumb_w, thumb_h = _matched_preview_size(face)
        if thumb_w + 18 <= x2 - x1 and thumb_h + 8 <= y2 - y1:
            thumb = cv2.resize(face.reference_crop, (thumb_w, thumb_h),
                               interpolation=cv2.INTER_AREA)
            tx, ty = x2 - thumb_w - 4, y1 + 4
            canvas[ty:ty + thumb_h, tx:tx + thumb_w] = thumb
            cv2.rectangle(canvas, (tx, ty), (tx + thumb_w, ty + thumb_h),
                          _face_color(face, mode), 1)
            text_width -= thumb_w + 4
    if image_first and show_thumbnails and face.crop is not None:
        parsing_pair = mode == 'FD' and second is not None
        if parsing_pair:
            images = [face.crop, face.mask]
            visualization = face.extras.get('parsing_visualization')
            if visualization is not None:
                images.append(visualization)
            labels = ['FACE', 'MASKED', 'SEGMENT'][:len(images)]
        else:
            images = [face.crop] + ([second] if second is not None else [])
            labels = []
        width_limit = (x2 - x1 - 16 - 10 * (len(images) - 1)) // len(images)
        height_limit = y2 - y1 - len(lines) * line_step - (64 if parsing_pair else 46)
        tile_size = min(160, width_limit, height_limit)
        if tile_size >= 32:
            tile_y = y1 + 8
            for index, image in enumerate(images):
                tile_x = x1 + 8 + index * (tile_size + 10)
                _tile(canvas, image, tile_x, tile_y, tile_size, '')
                if labels:
                    _text(canvas, labels[index], tile_x,
                          tile_y + tile_size + 16, 0.38,
                          max_width=tile_size, thickness=1)
            text_top = tile_y + tile_size + (40 if parsing_pair else 22)
    if inline_parsing and face.crop is not None and face.mask is not None:
        images = [face.crop, face.mask]
        visualization = face.extras.get('parsing_visualization')
        if visualization is not None:
            images.append(visualization)
        labels = ['FACE', 'MASK', 'SEG'][:len(images)]
        tile_y = y1 + 10 + len(lines) * line_step
        tile_size = min(72,
                        (x2 - x1 - 14 - 8 * (len(images) - 1)) // len(images),
                        y2 - tile_y - 18)
        if tile_size >= 32 and tile_y + tile_size + 18 <= y2:
            for index, image in enumerate(images):
                tile_x = x1 + 7 + index * (tile_size + 8)
                _tile(canvas, image, tile_x, tile_y, tile_size, '')
                _text(canvas, labels[index], tile_x,
                      tile_y + tile_size + 14, 0.34,
                      max_width=tile_size, thickness=1)
    for index, line in enumerate(lines):
        baseline = text_top + index * line_step
        if baseline > y2 - 5:
            break
        _text(canvas, line, x1 + 7, baseline,
              min(0.45, 0.45 * line_step / 17),
              max_width=text_width, thickness=1)


def _draw_reference_gallery(canvas, target_groups, source_height, output_width):
    if not target_groups:
        return
    group_width = output_width // len(target_groups)
    thumb_size = max(8, min(68, group_width - 20))
    pitch = thumb_size + 18
    row_height = thumb_size + 36
    for group_idx, group in enumerate(target_groups):
        gx = group_idx * group_width
        _text(canvas, f"{group['name']} ({len(group['faces'])} references)",
              gx + 10, source_height + 29, 0.68, max_width=group_width - 20)
        cols = max(1, (group_width - 16) // pitch)
        for ref_idx, image in enumerate(group['faces']):
            x = gx + 10 + (ref_idx % cols) * pitch
            y = source_height + 43 + (ref_idx // cols) * row_height
            _tile(canvas, image, x, y, thumb_size, str(ref_idx + 1))


def _matched_preview_size(face):
    x1, y1, x2, y2 = face.bbox
    return (max(52, ceil(max(0, x2 - x1) / 2)),
            max(52, ceil(max(0, y2 - y1) / 2)))


def _with_box_status(canvas, status):
    height, width = canvas.shape[:2]
    output = np.full((height + BOX_STATUS_HEIGHT, width, 3),
                     (20, 25, 34), dtype=np.uint8)
    output[BOX_STATUS_HEIGHT:] = canvas
    _text(output, status, 10, 28, 0.61, max_width=width - 20)
    return output


def render_face_frame(frame, result, mode, layout, target_groups=None,
                      infer_interval=1, history=None, result_layout='panel',
                      show_gaze=False, show_landmarks=False,
                      head_pose_enabled=False):
    """Render either frame-only result cards or the attached result panel."""
    h, w = frame.shape[:2]
    out_w, out_h, side_width, gallery_height = layout
    if result_layout == 'box':
        content_h = out_h - BOX_STATUS_HEIGHT
        if content_h == h:
            canvas = frame.copy()
        else:
            canvas = np.full((content_h, out_w, 3), (20, 25, 34), dtype=np.uint8)
            canvas[:h, :w] = frame
    else:
        canvas = np.full((out_h, out_w, 3), (20, 25, 34), dtype=np.uint8)
        canvas[:h, :w] = frame
    fps = 1 / result.inference_seconds if result.inference_seconds > 0 else 0.0
    tracking = result.tracking
    status = (f'{mode}  {len(result.faces)} faces  {fps:.1f} infer FPS  '
              f'track {"ON" if tracking else "OFF"}')
    if result_layout == 'panel':
        _text(canvas, status, w + 10, 31, 0.61, max_width=side_width - 20)
    for index, face in enumerate(result.faces, 1):
        color = _face_color(face, mode)
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        if not (head_pose_enabled or head_pose_box_points(face) is not None):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        draw_head_pose_box(canvas[:h, :w], face)
        points = np.asarray(face.landmarks, dtype=np.float32).reshape(-1, 2)
        if show_landmarks:
            for point in points[:5]:
                if np.all(np.isfinite(point)):
                    cv2.circle(canvas, tuple(np.rint(point).astype(int)), 3,
                               (255, 255, 255), -1, cv2.LINE_AA)
        pitch, yaw = face.attributes.get('pitch'), face.attributes.get('yaw')
        if (show_gaze and len(points) >= 2 and pitch is not None and yaw is not None
                and isfinite(pitch) and isfinite(yaw)):
            origin = np.mean(points[:2], axis=0)
            length = max(16, min(60, 0.7 * max(x2 - x1, y2 - y1)))
            endpoint = origin + (length * sin(yaw), -length * sin(pitch))
            start = tuple(np.rint(origin).astype(int))
            end = tuple(np.rint(endpoint).astype(int))
            if np.linalg.norm(endpoint - origin) >= 4:
                cv2.arrowedLine(canvas, start, end, (0, 255, 255), 2,
                                cv2.LINE_AA, tipLength=0.25)
            else:
                cv2.circle(canvas, start, 5, (0, 255, 255), 2, cv2.LINE_AA)
        face_id = face.track_id if face.track_id is not None else index
        if (not (head_pose_enabled or head_pose_box_points(face) is not None)
                and (mode == 'FS' or result.tracking or _fd_has_results(face))):
            tag = f'#{face_id}'
            cv2.rectangle(canvas, (max(0, x1), max(0, y1 - 28)),
                          (max(0, x1) + 54, max(0, y1)), color, -1)
            _text(canvas, tag, max(0, x1) + 4, max(18, y1 - 6), 0.6)

    if result_layout == 'box':
        if mode == 'FD' and not any(_fd_has_results(face) for face in result.faces):
            return _with_box_status(canvas, status)
        card_lines = _fd_card_lines if mode == 'FD' else _fs_card_lines
        eligible = [(index, face) for index, face in enumerate(result.faces, 1)
                    if mode == 'FS' or _fd_has_results(face)]
        faces = [face for _, face in eligible]
        full_cards = [card_lines(face, face.track_id or index)
                      for index, face in eligible]
        brief_cards = [_brief_card_lines(face, face.track_id or index, mode)
                       for index, face in eligible]
        def card_size(face, lines, brief=False):
            if mode == 'FS' and face.reference_crop is not None:
                thumb_w, thumb_h = _matched_preview_size(face)
                text_width = (min(135, max(90, w // 4)) if brief else
                              min(185, max(110, w // 3)))
                return (text_width + thumb_w + 18,
                        max(thumb_h + 8, 48 if brief else 12 + len(lines) * 16))
            if mode == 'FD' and face.mask is not None:
                tile_count = (3 if face.extras.get('parsing_visualization')
                              is not None else 2)
                tile_size = (min(58, max(32, w // 10)) if tile_count == 3
                             else min(72, max(32, w // 8)))
                width = (min(135, max(90, w // 4)) if brief else
                         min(185, max(110, w // 3)))
                return (max(width, tile_count * tile_size +
                            (tile_count - 1) * 8 + 14),
                        28 + len(lines) * 17 + tile_size)
            if brief:
                return min(135, max(90, w // 4)), 48
            return min(185, max(110, w // 3)), 12 + len(lines) * 16

        variants = (
            (full_cards, [card_size(face, lines)
                          for face, lines in zip(faces, full_cards)]),
            (brief_cards, [card_size(face, lines, brief=True)
                           for face, lines in zip(faces, brief_cards)]),
        )
        best = ([], [], [])
        indices = list(range(len(faces)))
        orders = (indices, list(reversed(indices)),
                  sorted(indices, key=lambda index:
                         (faces[index].bbox[2] - faces[index].bbox[0])
                         * (faces[index].bbox[3] - faces[index].bbox[1])))
        for cards, sizes in variants:
            complete = False
            for order in orders:
                assigned = plan_fd_card_positions(
                    [faces[index] for index in order],
                    [sizes[index] for index in order], (w, h), w)
                positions = [None] * len(faces)
                for index, rect in zip(order, assigned):
                    positions[index] = rect
                if sum(rect is not None for rect in positions) > sum(
                        rect is not None for rect in best[2]):
                    best = (cards, sizes, positions)
                if all(rect is not None for rect in positions):
                    best = (cards, sizes, positions)
                    complete = True
                    break
            if complete:
                break
        cards, _, positions = best
        for face, lines, rect in zip(faces, cards, positions):
            if rect is not None:
                _draw_result_card(canvas, face, lines, rect, mode,
                                  show_thumbnails=False,
                                  inline_reference=(mode == 'FS' and
                                                    face.reference_crop is not None),
                                  inline_parsing=(mode == 'FD' and face.mask is not None))
        _draw_reference_gallery(canvas, target_groups, h, out_w)
        return _with_box_status(canvas, status)

    # Adaptive grid in the side rail: every face gets a disjoint cell.
    count = len(result.faces)
    if count:
        rail_top = 48
        rail_height = max(1, h - rail_top)
        cols = (count if mode == 'FD' and count <= 4
                else max(1, ceil(sqrt(count * side_width / rail_height))))
        rows = ceil(count / cols)
        cell_w, cell_h = side_width // cols, rail_height // rows
        for index, face in enumerate(result.faces):
            col, row = index % cols, index // cols
            x, y = w + col * cell_w, rail_top + row * cell_h
            if cell_w < 8 or cell_h < 8:
                continue
            face_id = face.track_id if face.track_id is not None else index + 1
            lines = (_fd_card_lines(face, face_id) if mode == 'FD'
                     else _fs_card_lines(face, face_id))
            _draw_result_card(canvas, face, lines,
                              (x + 3, y + 3, x + cell_w - 4, y + cell_h - 4),
                              mode, image_first=True)
    else:
        _text(canvas, 'No faces detected', w + 16, 83, 0.8)

    _draw_reference_gallery(canvas, target_groups, h, out_w)
    if history:
        top = h + gallery_height
        _text(canvas, 'RECENT FACES', 10, top + 27, 0.65)
        for index, (image, frame_index) in enumerate(history):
            x = 170 + index * 86
            if x + 68 > out_w:
                break
            _tile(canvas, image, x, top + 8, 68, str(frame_index))
    return canvas
