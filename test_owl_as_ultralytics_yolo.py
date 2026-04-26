#!/usr/bin/env python3

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from ultralytics.engine.results import Results


class OWLYOLO:

    def __init__(self,
                 model_path="google/owlv2-base-patch16-ensemble",
                 query_texts=None,
                 query_images=None,
                 conf=0.1,
                 min_area=0,
                 max_area=float('inf'),
                 rois=None,
                 strict_roi=False,
                 device=None):
        """
        Initialize with persistent defaults.
        """
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")

        # Load Model/Processor
        self.processor = Owlv2Processor.from_pretrained(model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(
            self.device)
        self.model.eval()

        # Save Defaults
        self.query_texts = query_texts
        self.query_images = query_images
        self.conf = conf
        self.min_area = min_area
        self.max_area = max_area
        self.rois = rois
        self.strict_roi = strict_roi

    def _check_roi(self, box, rois, strict):
        if not rois: return True
        bx1, by1, bx2, by2 = box
        for roi in rois:
            rx1, ry1, rx2, ry2 = roi
            if strict:
                if bx1 >= rx1 and by1 >= ry1 and bx2 <= rx2 and by2 <= ry2:
                    return True
            else:
                if max(bx1, rx1) < min(bx2, rx2) and max(by1, ry1) < min(
                        by2, ry2):
                    return True
        return False

    def _filter_result(self,
                       results_raw,
                       min_area=None,
                       max_area=None,
                       rois=None,
                       strict_roi=None):

        mi_a = min_area if min_area is not None else self.min_area
        ma_a = max_area if max_area is not None else self.max_area
        r = rois if rois is not None else self.rois
        s_r = strict_roi if strict_roi is not None else self.strict_roi

        final_boxes, final_scores, final_labels = [], [], []
        for i, (box, score) in enumerate(
                zip(results_raw["boxes"], results_raw["scores"])):

            coords = box.tolist()
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])

            if not (mi_a <= area <= ma_a): continue
            if not self._check_roi(coords, r, s_r): continue

            final_boxes.append(coords)
            final_scores.append(score.item())
            if results_raw['labels'] is not None:
                final_labels.append(results_raw['labels'][i].item())
            else:
                final_labels.append(0)

        return final_boxes, final_scores, final_labels

    def _convert_to_ultralytics_result(self, image_pil, final_boxes,
                                       final_scores, final_labels, label_map):

        # Pack into Results (Convert back to BGR for Ultralytics)
        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        if final_boxes:
            det = torch.cat([
                torch.tensor(final_boxes),
                torch.tensor(final_scores).unsqueeze(1),
                torch.tensor(final_labels).unsqueeze(1)
            ],
                            dim=1)
        else:
            det = torch.empty((0, 6))

        return [
            Results(orig_img=img_bgr,
                    path="inference.jpg",
                    names=label_map,
                    boxes=det)
        ]

    def __call__(self,
                 source,
                 query_texts=None,
                 query_images=None,
                 conf=None,
                 min_area=None,
                 max_area=None,
                 rois=None,
                 strict_roi=None):

        # 1. Resolve Parameters: Use call-time args if provided, else use init defaults
        q = query_texts if query_texts is not None else self.query_texts
        q_img = query_images if query_images is not None else self.query_images
        c = conf if conf is not None else self.conf

        # Early stop
        if q is None and q_img is None:
            raise ValueError(
                "Queries must be provided either in __init__ or __call__.")

        # 2. Handle Image Colorspaces
        if isinstance(source, str):
            image_pil = Image.open(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            # Convert BGR (OpenCV) to RGB (HuggingFace)
            image_pil = Image.fromarray(cv2.cvtColor(source,
                                                     cv2.COLOR_BGR2RGB))
        else:
            image_pil = source
        width, height = image_pil.size

        # 3. Inference

        # 3-1. Inference w/ query text
        if q is not None:
            inputs = self.processor(text=[q],
                                    images=image_pil,
                                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            target_sizes = torch.Tensor([[height, width]]).to(self.device)
            results_raw = self.processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=c)[0]
            label_map = {i: label for i, label in enumerate(q)}

        # 3-2. Inference w/ query image
        elif q_img is not None:
            inputs = self.processor(images=image_pil,
                                    query_images=[q_img],
                                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.image_guided_detection(**inputs)
            target_sizes = torch.Tensor([[height, width]]).to(self.device)
            results_raw = self.processor.post_process_image_guided_detection(
                outputs,
                target_sizes=target_sizes.repeat(len(q_img), 1),
                nms_threshold=0.0,
                threshold=c)[0]
            if results_raw['labels'] is None:
                label_map = {0: 'target'}

        # 4. Filter result
        final_boxes, final_scores, final_labels = \
                self._filter_result(results_raw,
                        min_area=min_area,
                        max_area=max_area,
                        rois=rois,
                        strict_roi=strict_roi)

        # 5. Convert result to Ultralytics result
        return self._convert_to_ultralytics_result(image_pil, final_boxes,
                                                   final_scores, final_labels,
                                                   label_map)

    def save_local(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)


def visualize_ultralytics_result(results):
    # See the raw filtered tensors
    boxes = results[0].boxes.xyxy.cpu().tolist()
    cls_ids = results[0].boxes.cls.int().cpu().tolist()
    confs = results[0].boxes.conf.cpu().tolist()
    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        x1, y1, x2, y2 = map(int, box)
        object_name = 'Query'
        if query_texts is not None:
            object_name = f'{query_texts[cls_id]}'
        print(f'- {object_name} {conf:.2f} {x1} {y1} {x2} {y2} ')
    # Use standard Ultralytics methods
    canvas = results[0].plot()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="OWLv2 Detection Script")
    # 1. input_image_paths: Required, one or more
    parser.add_argument('-i',
                        '--input_image_paths',
                        nargs='+',
                        required=True,
                        help="Path(s) to the target images to be searched.")
    # 2. query_texts: Optional list, defaults to None
    parser.add_argument(
        '-qt',
        '--query_texts',
        nargs='+',
        default=None,
        help="Text prompts for zero-shot detection (e.g., 'a cat' 'a remote')."
    )
    # 3. query_image_paths: Optional list, defaults to None
    parser.add_argument(
        '-qi',
        '--query_image_paths',
        nargs='+',
        default=None,
        help="Path(s) to query images for image-guided detection.")
    # 4. ROIs: Optional list of 'x1,y1,x2,y2' strings
    parser.add_argument('--rois',
                        nargs='+',
                        default=None,
                        help="Regions of Interest in 'x1,y1,x2,y2' format.")
    # 5. Confidence Threshold: Float, defaults to 0.1
    parser.add_argument(
        '-c',
        '--conf',
        type=float,
        default=0.1,
        help="Confidence threshold for filtering detections (default: 0.1).")
    args = parser.parse_args()

    # Input
    input_image_paths = args.input_image_paths

    # Query Text
    query_texts = args.query_texts

    # Query Image
    query_image_paths = args.query_image_paths
    query_images = None
    if query_image_paths is not None:
        query_images = [
            Image.open(image_path) for image_path in query_image_paths
        ]

    # ROI
    if args.rois is not None:
        ROIs = []
        for roi in args.rois:
            coord = roi.split(',')
            if len(coord) < 4:
                continue
            coord = list(map(int, coord))[:4]
            ROIs.append(coord)

    # Conf
    conf = args.conf

    # Init model
    detector = OWLYOLO(query_texts=query_texts,
                       query_images=query_images,
                       rois=ROIs,
                       conf=conf)

    # Run like Ultralytics YOLO
    for input_image_path in input_image_paths:
        print(f'Input: {input_image_path}')
        image = cv2.imread(input_image_path)
        results = detector(image, strict_roi=True)
        canvas = visualize_ultralytics_result(results)
        plt.imshow(canvas)
        plt.show()
