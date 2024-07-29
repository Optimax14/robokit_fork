import os 
import numpy as np
from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import overlay_masks
from robokit.perception import SegmentAnythingPredictor
import time

input_directory_path = '/Path/to/INPUT/folder'
output_directory_path = '/Path/to/OUTPUT/folder'

def main(local_path, output_path):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    images = [os.path.join(local_path, f) for f in os.listdir(local_path)]

    try:
        for image_path in images:
            print(f"image path -- {image_path}")
            logging.info("Initialize object detectors")
            SAM = SegmentAnythingPredictor()
    
            logging.info("Open the image and convert to RGB format")
            image_pil = PILImg.open(image_path).convert("RGB")
            w, h = image_pil.size
    
            logging.info("SAM prediction")
            image_pil_bboxes, masks = SAM.predict(image_pil, prompt_bboxes=np.array([0, 0, w, h]))
            # if prompt_bboxes is None, SAM will generate masks for the entire image: todo not yet complete
            # image_pil_bboxes, masks = SAM.predict(image_pil, prompt_bboxes=None)
    
            overlayed_image = overlay_masks(image_pil, masks)

            # Save the overlayed image to the output directory
            output_image_path = os.path.join(output_path, os.path.basename(image_path))
            overlayed_image.save(output_image_path)
            print(f"Saved overlayed image to {output_image_path}")

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main(input_directory_path, output_directory_path)
    
