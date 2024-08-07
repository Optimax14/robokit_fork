#!/usr/bin/env python

# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.


# 2024 modified by Itay Kadosh.
# Added filtering functionality, taking images from a local directory, and publishing results to be listened to through a subscriber (test_listener.py)
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas


from absl import app, logging
from PIL import (Image as PILImg, ImageDraw)
from robokit.utils import annotate, overlay_masks, draw_mask, save_mask, filter
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import os
import time
import numpy as np
import torch
import sys


def main(argv): 
    root_dir = argv
    input_path = os.path.join(root_dir, "color")
    output_path = os.path.join(root_dir, "segments")

    images = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    text_prompt = 'chair . table . door .'

    try:
        logging.info("Initialize object detectors")
        gdino = GroundingDINOObjectPredictor()
        SAM = SegmentAnythingPredictor()

        for image_path in images:

            logging.info("Open the image and convert to RGB format")
            image_pil = PILImg.open(image_path).convert("RGB")

            logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
            bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt, 0.4, 0.4)

            bboxes, gdino_conf, phrases ,flag = filter(bboxes, gdino_conf, phrases, 1, 0.8, 0.8, 0.8, 0.01)
            if flag:
                continue

            logging.info("GDINO post processing")
            w, h = image_pil.size  # Get image width and height
            # Scale bounding boxes to match the original image size
            image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)
            # print(image_pil_bboxes)

            logging.info("SAM prediction")
            image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)

            logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
            bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)

            segments_color_path = os.path.join(root_dir, "segments_color")
            os.makedirs(segments_color_path, exist_ok=True)

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            annotated_image_path = os.path.join(segments_color_path, f"{image_name}.png")
            bbox_annotated_pil.save(annotated_image_path)

            save_mask(masks, output_path, image_path, phrases)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main(sys.argv[1])
