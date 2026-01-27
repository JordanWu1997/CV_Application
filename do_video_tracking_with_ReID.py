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
# |_|\_\  |_| |_|  Datetime: 2026-01-27 22:00:10             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, cosine
from tqdm import tqdm
from ultralytics import YOLO

import torchreid


class GlobalReIDTracker:

    def __init__(self,
                 extractor,
                 max_age=30,
                 spatial_gate=0.5,
                 threshold=0.80):

        self.extractor = extractor
        self.threshold = threshold
        self.max_age = max_age  # Seconds to keep a person in memory
        self.spatial_gate = spatial_gate  # Max % of frame width a person can "jump"

        # {gid: {'feat': vector, 'color': color, 'last_seen': timestamp, 'last_pos': (cx, cy)}}
        self.global_gallery = {}
        self.id_map = {}
        self.next_global_id = 1

    def get_color(self, gid):
        np.random.seed(gid)
        return tuple(map(int, np.random.randint(0, 255, 3)))

    def _get_centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _cleanup_gallery(self):
        """Removes IDs that haven't been seen for max_age seconds."""
        current_time = time.time()
        expired_ids = [
            gid for gid, data in self.global_gallery.items()
            if current_time - data['last_seen'] > self.max_age
        ]
        for gid in expired_ids:
            del self.global_gallery[gid]
            # Clean up id_map entries pointing to this gid
            self.id_map = {k: v for k, v in self.id_map.items() if v != gid}

    def process_frame_results(self, frame, results):
        self._cleanup_gallery()
        h, w = frame.shape[:2]

        if results[0].boxes.id is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        yolo_ids = results[0].boxes.id.cpu().numpy().astype(int)

        current_frame_data = []
        for box, yid in zip(boxes, yolo_ids):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                feat = self.extractor(crop).cpu().detach().numpy().flatten()
                current_frame_data.append({
                    'yid':
                    yid,
                    'feat':
                    feat,
                    'box': [x1, y1, x2, y2],
                    'centroid':
                    self._get_centroid([x1, y1, x2, y2]),
                    'assigned_gid':
                    None
                })

        if not self.global_gallery:
            return self._initialize_new_tracks(frame, current_frame_data)

        # 1. Prepare Distance Matrix
        gids_in_gallery = list(self.global_gallery.keys())
        gallery_feats = np.array(
            [self.global_gallery[g]['feat'] for g in gids_in_gallery])
        new_feats = np.array([d['feat'] for d in current_frame_data])

        dist_matrix = cdist(new_feats, gallery_feats, metric='cosine')

        # 2. Apply Spatial Gating
        # If the person moved more than 'spatial_gate' of the screen width,
        # we make the match impossible (dist = 1.0)
        for i, det in enumerate(current_frame_data):
            for j, gid in enumerate(gids_in_gallery):
                last_pos = self.global_gallery[gid].get('last_pos')
                if last_pos:
                    # Calculate Euclidean distance between centroids normalized by image width
                    spatial_dist = np.linalg.norm(
                        np.array(det['centroid']) - np.array(last_pos)) / w
                    if spatial_dist > self.spatial_gate:
                        dist_matrix[i, j] = 1.0

        # 3. Hungarian Matching
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        assigned_rows = set()
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < (1 - self.threshold):
                gid = gids_in_gallery[c]
                yid = current_frame_data[r]['yid']

                # Update Gallery
                self.global_gallery[gid].update({
                    'feat':
                    0.9 * self.global_gallery[gid]['feat'] +
                    0.1 * current_frame_data[r]['feat'],
                    'last_seen':
                    time.time(),
                    'last_pos':
                    current_frame_data[r]['centroid']
                })
                current_frame_data[r]['assigned_gid'] = gid
                self.id_map[yid] = gid
                assigned_rows.add(r)

        # 4. Initialize New Tracks
        for i, data in enumerate(current_frame_data):
            if i not in assigned_rows:
                new_gid = self.next_global_id
                self.global_gallery[new_gid] = {
                    'feat': data['feat'],
                    'color': self.get_color(new_gid),
                    'last_seen': time.time(),
                    'last_pos': data['centroid']
                }
                self.id_map[data['yid']] = new_gid
                data['assigned_gid'] = new_gid
                self.next_global_id += 1

        return self._draw(frame, current_frame_data)

    def _initialize_new_tracks(self, frame, current_frame_data):
        for data in current_frame_data:
            new_gid = self.next_global_id
            self.global_gallery[new_gid] = {
                'feat': data['feat'],
                'color': self.get_color(new_gid),
                'last_seen': time.time(),
                'last_pos': data['centroid']
            }
            self.id_map[data['yid']] = new_gid
            data['assigned_gid'] = new_gid
            self.next_global_id += 1
        return self._draw(frame, current_frame_data)

    def _draw(self, frame, current_frame_data):
        for data in current_frame_data:
            gid = data.get('assigned_gid')
            x1, y1, x2, y2 = data['box']
            color = self.global_gallery[gid]['color']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"G-ID: {gid}", (x1, y1 + 15), 0, 0.6, color, 2)
        return frame


if __name__ == '__main__':

    # Load YOLO model
    yolo_model = YOLO('./weights/yolo11n.pt')

    # Load torchreid model
    model_name = 'osnet_ain_x1_0'
    model_path = './models/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
    extractor = torchreid.utils.FeatureExtractor(model_name=model_name,
                                                 model_path=model_path,
                                                 device='cuda')

    input_video_paths = sys.argv[1:]
    for input_video_path in input_video_paths:

        # Init Re-ID
        reid_system = GlobalReIDTracker(extractor,
                                        threshold=0.4,
                                        spatial_gate=1.0,
                                        max_age=300)

        # Load input video
        input_video_cap = cv2.VideoCapture(input_video_path)
        input_video_dir = os.path.dirname(input_video_path)
        input_video_name, ext = os.path.splitext(
            os.path.basename(input_video_path))
        FPS = input_video_cap.get(cv2.CAP_PROP_FPS)
        total_frame = int(input_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(input_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(input_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Load output video output writer
        output_video_path = f'{input_video_dir}/{input_video_name}_ReID{ext}'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS,
                                              (frame_width, frame_height))

        # Main
        progress_bar = tqdm(total=total_frame)
        while input_video_cap.isOpened():
            progress_bar.update(1)

            ret, frame = input_video_cap.read()
            if not ret:
                break

            # Detect and track
            results = yolo_model.track(frame,
                                       persist=True,
                                       verbose=False,
                                       classes=[0],
                                       conf=0.5,
                                       tracker='bytetrack.yaml')

            # Update gid and visualize
            canvas = reid_system.process_frame_results(frame, results)
            output_video_writer.write(canvas)

            # Resize for display
            canvas = cv2.resize(canvas, (1280, 720))
            cv2.imshow('Multi-Camera ReID', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        progress_bar.close()
        input_video_cap.release()
        cv2.destroyAllWindows()
        output_video_writer.release()
