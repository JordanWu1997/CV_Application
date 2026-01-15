#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import time
from random import randint

import cv2
import numpy as np
from ultralytics import SAM, FastSAM

if __name__ == '__main__':

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Load models
    start = time.time()
    # model = SAM('./weights/sam2_t.pt')
    # model = SAM('./weights/mobile_sam.pt')
    model = FastSAM('./weights/FastSAM-s.pt')
    model.info()
    print(f'[INFO] Loading model took {time.time()-start:.3f} sec')

    # Inference
    start = time.time()
    results = model.predict(image)
    print(f'[INFO] Inferencing took {time.time()-start:.3f} sec')

    # Visualization (segmentation)
    result_image = image.copy()
    masks = results[0].masks.data.cpu().tolist()
    for i, mask in enumerate(masks):
        mask = np.array(mask, dtype=bool)
        canvas = np.zeros_like(mask, dtype=np.uint8)
        canvas = np.array([canvas] * 3)
        canvas = np.transpose(canvas, (1, 2, 0))
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        canvas[mask] = color
        canvas = cv2.resize(canvas, (width, height))
        result_image = cv2.addWeighted(result_image, 1, canvas, 0.5, 0)
    cv2.imwrite('output_seg.png', result_image)

    # Visualization (bounding box)
    result_image = image.copy()
    bboxes = results[0].boxes.xywh.cpu().tolist()
    print(f'[INFO] BBoxes: {bboxes}')
    for bbox in bboxes:
        x, y, w, h = bbox
        x1, x2 = int(x - w / 2), int(w + w / 2)
        y1, y2 = int(y - h / 2), int(y + h / 2)
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), \
                                     color, 1)
    cv2.imwrite('output_box.png', result_image)
