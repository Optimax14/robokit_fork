# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import os
import time


directory_path = '/home/itaykadosh/Desktop/2024-07-17_16-36-17/color'
output_directory = '/home/itaykadosh/Desktop/2024-07-17_16-36-17/output_gdino'

def main(local_path, output_path):
    # Path to the input image

    images = [os.path.join(local_path, f) for f in os.listdir(local_path)]
    text_prompt =  'table'

    try:
        logging.info("Initialize object detectors")
        gdino = GroundingDINOObjectPredictor()
        # SAM = SegmentAnythingPredictor()

        for image_path in images:
            logging.info("Open the image and convert to RGB format")
            image_pil = PILImg.open(image_path).convert("RGB")
            
            logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
            bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

            logging.info("GDINO post processing")
            w, h = image_pil.size # Get image width and height 
            # Scale bounding boxes to match the original image size
            image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

            # logging.info("SAM prediction")
            # image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)

            logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
            bbox_annotated_pil = annotate(image_pil, image_pil_bboxes, gdino_conf, phrases)


            bbox_annotated_pil.show()
            time.sleep(5)
            output_image_path = os.path.join(output_path, os.path.basename(image_path))
            # bbox_annotated_pil.save(output_image_path)
            print(f"Saved overlayed image to {output_image_path}")

            # bbox_annotated_pil.show()


    except Exception as e:
        # Handle unexpected errorsS
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with the input image path
    # app.run(main, ['imgs/color-000078.png'])
    # app.run(main, ['imgs/color-000019.png'])
    # app.run(main, ['imgs/irvl-clutter-test.png'])
    main(directory_path, output_directory)
