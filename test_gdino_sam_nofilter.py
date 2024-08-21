#!/usr/bin/env python

# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

# 2024 modified by Itay Kadosh.
# Added filtering functionality, taking images from a local directory, and saving masks into specific subfolders
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas

from absl import app, logging
from PIL import (Image as PILImg, ImageDraw)
from robokit.utils import annotate, overlay_masks, draw_mask, save_mask, filter
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import os
import sys

def main(argv): 
    root_dir = argv
    input_path = os.path.join(root_dir, "color")
    
    # List of prompts
    # prompts = [
    #     "door", "person", "shelves", "cabinet", "exit sign", "fire extinguisher", "open door", 
    #     "closed door", "glass", "mirror", "Trash bin", "bin", "hallway", 
    #     "water fountain", "filter", "bottle", "cup", "mug", "bench", 
    #     "laptop", "bag"
    # ]
    prompts = [
        "open door"
    ]

    images = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    images.sort()

    try:
        print("Initialize object detectors")
        print("here")
        gdino = GroundingDINOObjectPredictor()
        SAM = SegmentAnythingPredictor()

        for prompt in prompts:
            print(f"Processing prompt: {prompt}")

            
            prompt_folder = os.path.join(os.path.join(root_dir, prompt), "_nofilter")
            print(f"Processing prompt: {prompt_folder}")
            os.makedirs(prompt_folder, exist_ok=True)
            output_path = os.path.join(prompt_folder, "segments")

            for image_path in images:
                try:
                    print(f"Processing image: {image_path}")
                    print(f"Processing image: {image_path}")
                    image_pil = PILImg.open(image_path).convert("RGB")

                    print("GDINO: Predict bounding boxes, phrases, and confidence scores")
                    bboxes, phrases, gdino_conf = gdino.predict(image_pil, prompt, 0.55, 0.55)

                    # bboxes, gdino_conf, phrases, flag = filter(bboxes, gdino_conf, phrases, 1, 0.8, 0.8, 0.8, 0.01, False)
                    # if flag:
                    #     continue
                    if bboxes.size(dim=1)==0:
                        continue
                    print("GDINO post processing")
                    w, h = image_pil.size  # Get image width and height
                    image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)
                    
                    print("SAM prediction")
                    image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)
                    
                    print("Annotate and save the image with bounding boxes, confidence scores, and labels")
                    bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)

                    segments_color_path = os.path.join(prompt_folder, "segments_color")
                    os.makedirs(segments_color_path, exist_ok=True)

                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    annotated_image_path = os.path.join(segments_color_path, f"{image_name}.png")
                    bbox_annotated_pil.save(annotated_image_path)

                    save_mask(masks, output_path, image_path, phrases)
                except Exception as e:
                    continue
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main(sys.argv[1])
