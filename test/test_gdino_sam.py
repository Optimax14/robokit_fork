# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

from absl import app, logging
from PIL import (Image as PILImg, ImageDraw)
from robokit.utils import annotate, overlay_masks, draw_mask
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import os 
import time

input_directory_path = '/Path/to/INPUT/folder'

def filter(bboxes, conf_list, conf_bound, yVal ,precentWidth = 0.5, precentHeight = 0.5):
    print("PRE FILTER")
    print(conf_list.size(dim=0))
    print(bboxes)            
    for conf in conf_list:
        print(conf)

    # bboxes is a tensor of size [n,4] where every row is structured by center x center y width(%OfScreen) height(%OfScreen)
    # Filters out boxes that are the more than half of the image in size
    if conf_list.size(dim=0) >= 1:
        c1 = bboxes[:,3] <= precentHeight # taller than 50% (Default)
        c2 = bboxes[:,2] <= precentWidth # wider than 50% (Default)
        c3 = bboxes[:,1] <= yVal
        mask = c1 & c2 & c3
        bboxes = bboxes[mask]
        conf_list = conf_list[mask]

    # Filters out images with detections at all
    if conf_list.size(dim=0) == 0:
        return bboxes, conf_list, True
    
    print("POST FILTER")
    print(conf_list.size(dim=0))
    print(bboxes)            
    for conf in conf_list:
        print(conf)

    # Creates an upper bound for confidence
    if any(conf >= conf_bound for conf in conf_list):
        return bboxes, conf_list, True
    
    return bboxes, conf_list, False


def main(local_path):
    # Path to the input image
    
    images = [os.path.join(local_path, f) for f in os.listdir(local_path)]
    text_prompt =  'table'


    try:
        logging.info("Initialize object detectors")
        gdino = GroundingDINOObjectPredictor()
        SAM = SegmentAnythingPredictor()
        for image_path in images:

            logging.info("Open the image and convert to RGB format")
            image_pil = PILImg.open(image_path).convert("RGB")
            
            logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
            bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt,0.4,0.4)

            bboxes, gdino_conf, flag = filter(bboxes,gdino_conf,0.6,0.8 ,0.8,0.8)
            if flag == True:
                continue

            logging.info("GDINO post processing")
            w, h = image_pil.size # Get image width and height 
            # Scale bounding boxes to match the original image size
            image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)
            print(image_pil_bboxes)

            logging.info("SAM prediction")
            image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)


            # prevents the detection of the entire screen as an object, additionally prevents noise on the floor from being detected  ** Size of screen is 640x480  
            if image_pil_bboxes[0,3].item() >= 400:
                continue 

            logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
            bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)

            # print(masks)
            # mask_image = PILImg.new('RGBA', image_pil.size, color=(0, 0, 0, 0))
            # mask_draw = ImageDraw.Draw(mask_image)
            # for mask in masks:
            #     draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)


            print("SUCCESSFUL ITERATION")
            bbox_annotated_pil.show()
            # mask_image.show()

            time.sleep(5)



    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with the input image path
    # app.run(main, ['imgs/color-000078.png'])
    # app.run(main, ['imgs/color-000019.png'])
    # app.run(main, ['imgs/irvl-clutter-test.png'])
    main(input_directory_path)
