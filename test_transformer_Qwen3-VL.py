#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import os
import sys
import textwrap
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from utils.utils import (put_chinese_text_to_canvas, resize_image_if_needed,
                         save_TF_model_to_local)


def build_messages(system_prompt: str,
                   user_prompt: str,
                   image_paths: list[str] | None = None,
                   max_image_size: int | None = None):
    messages = [{
        "role": "system",
        "content": [{
            "type": "text",
            "text": system_prompt
        }]
    }]

    user_content = []
    if image_paths:
        if max_image_size is not None:
            processed_paths = [
                resize_image_if_needed(path, max_image_size)
                for path in image_paths
            ]
        else:
            processed_paths = image_paths
        user_content.extend([{
            "type": "image",
            "min_pixels": 512 * 32 * 32,
            "max_pixels": 2048 * 32 * 32,
            "image": image_path
        } for image_path in processed_paths])
    user_content.append({"type": "text", "text": user_prompt})

    messages.append({
        "role": "user",
        "content": user_content,
    })

    return messages


def inference(model,
              processor,
              system_prompt: str,
              user_prompt: str,
              max_new_tokens: int = 1024,
              image_paths: list[str] | None = None,
              max_image_size: int | None = None):
    messages = build_messages(system_prompt, user_prompt, image_paths,
                              max_image_size)

    inputs = processor.apply_chat_template(messages,
                                           tokenize=True,
                                           add_generation_prompt=True,
                                           return_dict=True,
                                           return_tensors="pt")
    inputs = inputs.to(model.device)

    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return output_text[0]


def load_qwen3_VL_model(model="Qwen/Qwen3-VL-4B-Instruct",
                        local_model_dir='./models/qwen3-vl-4b-instruct'):

    if not os.path.isdir(local_model_dir):
        # Load model and save model to load
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            dtype="auto",
            device_map="auto",
            trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct",
                                                  trust_remote_code=True)
        save_TF_model_to_local(model, processor,
                               './models/qwen3-vl-4b-instruct')
    else:
        # Load saved model
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            local_model_dir, dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(local_model_dir)
    return model, processor


def main():

    # Load model
    model, processor = load_qwen3_VL_model()

    # Main
    system_prompt = """
    You are a helpful assistant that can answer questions and help with tasks.
    """
    user_prompt = "Read all the text in the image."
    input_image_paths = sys.argv[1:]
    for input_image_path in input_image_paths:
        # Inference
        print(f'[INFO] Input image path: {input_image_path}')
        output = inference(model,
                           processor,
                           system_prompt,
                           user_prompt,
                           image_paths=[input_image_path],
                           max_image_size=1536)
        print(f'{output}')
        print()

        # Visualization
        image = cv2.imread(input_image_path)
        canvas = 255 * np.ones_like(image)
        height, width, _ = canvas.shape
        font_size = max(18, int(0.25 * height / len(output.split('\n'))))
        canvas = put_chinese_text_to_canvas(canvas,
                                            output,
                                            top_left=(50, 50),
                                            font_size=font_size,
                                            font_path='./fonts/simfang.ttf')
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (255, 0, 0), 20,
                      cv2.LINE_AA)
        combined_view = np.hstack((image, canvas))
        combined_view = cv2.cvtColor(combined_view, cv2.COLOR_BGR2RGB)
        plt.imshow(combined_view)
        plt.show()


if __name__ == '__main__':
    main()
