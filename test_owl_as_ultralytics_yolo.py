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
                 queries=None,
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
        self.queries = queries
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

    def __call__(self,
                 source,
                 queries=None,
                 conf=None,
                 min_area=None,
                 max_area=None,
                 rois=None,
                 strict_roi=None):
        # 1. Resolve Parameters: Use call-time args if provided, else use init defaults
        q = queries if queries is not None else self.queries
        c = conf if conf is not None else self.conf
        mi_a = min_area if min_area is not None else self.min_area
        ma_a = max_area if max_area is not None else self.max_area
        r = rois if rois is not None else self.rois
        s_r = strict_roi if strict_roi is not None else self.strict_roi

        if q is None:
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
        inputs = self.processor(text=[q],
                                images=image_pil,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([[height, width]]).to(self.device)
        results_raw = self.processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes, threshold=c)[0]

        # 4. Filter
        final_boxes, final_scores, final_labels = [], [], []
        for box, score, label in zip(results_raw["boxes"],
                                     results_raw["scores"],
                                     results_raw["labels"]):
            coords = box.tolist()
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])

            if not (mi_a <= area <= ma_a): continue
            if not self._check_roi(coords, r, s_r): continue

            final_boxes.append(coords)
            final_scores.append(score.item())
            final_labels.append(label.item())

        # 5. Pack into Results (Convert back to BGR for Ultralytics)
        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        label_map = {i: label for i, label in enumerate(q)}

        if final_boxes:
            det = torch.cat([
                torch.tensor(final_boxes),
                torch.tensor(final_scores).unsqueeze(1),
                torch.tensor(final_labels).unsqueeze(1)
            ],
                            dim=1)
        else:
            det = torch.empty((0, 6))

        return Results(orig_img=img_bgr,
                       path="inference.jpg",
                       names=label_map,
                       boxes=det)

    def save_local(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)


if __name__ == "__main__":

    # Input
    input_image_paths = sys.argv[1:]
    queries = ["cat", "dog", "laptop"]
    ROIs = [[0, 0, 1000, 1000], [1500, 0, 2000, 1000]]

    # Init model
    detector = OWLYOLO(queries=queries, rois=ROIs, conf=0.25)

    # Run like Ultralytics YOLO
    for input_image_path in input_image_paths:
        print(f'Input: {input_image_path}')
        image = cv2.imread(input_image_path)
        results = detector(image, strict_roi=True)

        # See the raw filtered tensors
        boxes = results.boxes.xyxy.cpu().tolist()
        cls_ids = results.boxes.cls.int().cpu().tolist()
        confs = results.boxes.conf.cpu().tolist()
        for box, cls_id, conf in zip(boxes, cls_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            print(f'- {queries[cls_id]} {conf:.2f} {x1} {y1} {x2} {y2} ')

        # Use standard Ultralytics methods
        canvas = results.plot()
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        plt.imshow(canvas)
        plt.show()
