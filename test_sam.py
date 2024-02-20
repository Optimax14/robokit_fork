import numpy as np
from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import overlay_masks
from robokit.ObjDetection import SegmentAnythingPredictor

def main(argv):
    # Path to the input image
    image_path = argv[0]

    try:
        logging.info("Initialize object detectors")
        SAM = SegmentAnythingPredictor()

        logging.info("Open the image and convert to RGB format")
        image_pil = PILImg.open(image_path).convert("RGB")
        w, h =image_pil.size

        logging.info("SAM prediction")
        # if prompt_bboxes is None, SAM will generate masks for the entire image: todo not yet complete
        image_pil_bboxes, masks = SAM.predict(image_pil, prompt_bboxes=np.array([0,0,w//2,h//2]))
        logging.info(":")

        overlay_masks(image_pil,masks).show()


    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Run the main function with the input image path
    app.run(main, ['imgs/color-000078.png'])
    # app.run(main, ['imgs/color-000019.png'])