# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
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

    # Path to the input image
    image_path = argv[0]
    text_prompt =  'objects'

    try:
        logging.info("Initialize object detectors")
        gdino = GroundingDINOObjectPredictor()
        SAM = SegmentAnythingPredictor()

        logging.info("Open the image and convert to RGB format")
        image_pil = PILImg.open(image_path).convert("RGB")
        
        logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
        bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

        logging.info("GDINO post processing")
        w, h = image_pil.size # Get image width and height 
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        logging.info("SAM prediction")
        image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)

        logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)

        bbox_annotated_pil.show()

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with the input image path
    # app.run(main, ['imgs/color-000078.png'])
    # app.run(main, ['imgs/color-000019.png'])
    app.run(main, ['imgs/irvl-clutter-test.png'])
