#!/usr/bin/env python

# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.


# 2024 modified by Itay Kadosh.
# Added filtering functionality, taking images from a local directory, and publishing results to be listened to through a subscriber (test_listener.py)
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas


from absl import app, logging
from PIL import (Image as PILImg, ImageDraw)
from robokit.utils import annotate, overlay_masks, draw_mask
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import os
import time
import numpy as np
# import rospy
# from std_msgs.msg import Float32MultiArray, MultiArrayDimension

root_dir = '/home/itaykadosh/Desktop/datasets_robokit/2024-08-01_12-32-35'

def filter(bboxes, conf_list, phrases ,conf_bound, yVal, precentWidth=0.5, precentHeight=0.5, precentArea=0.05):
    print("PRE FILTER")
    print(conf_list.size(dim=0))
    #print(bboxes)
    #for conf in conf_list:
    #    print(conf)

    # Constants
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT
    MIN_BOX_AREA = precentArea * IMAGE_AREA

    # bboxes is a tensor of size [n,4] where every row is structured by center x center y width(%OfScreen) height(%OfScreen)
    # Filters out boxes that are the more than half of the image in size

    # if conf_list.size(dim=0) == 1:
    #     return bboxes, conf_list, True
    
    phrases_np = np.array(phrases)

    if conf_list.size(dim=0) >= 1:
        c1 = bboxes[:, 3] <= precentHeight  # taller than 50% (Default)
        c2 = bboxes[:, 2] <= precentWidth  # wider than 50% (Default)
        c3 = bboxes[:, 1] <= yVal
        
        # Calculate area of each bounding box and filter out those with area < 5% of total image
        box_areas = (bboxes[:, 2] * IMAGE_WIDTH) * (bboxes[:, 3] * IMAGE_HEIGHT)
        c4 = box_areas >= MIN_BOX_AREA
        
        mask = c1 & c2 & c3 & c4
        bboxes = bboxes[mask]
        conf_list = conf_list[mask]

        phrases_np = phrases_np[mask.cpu().numpy()]

    # Filters out images with detections at all
    if conf_list.size(dim=0) == 0:
        return bboxes, conf_list, phrases_np.tolist() ,True

    print("POST FILTER")
    print(conf_list.size(dim=0))
    #print(bboxes)
    #for conf in conf_list:
    #    print(conf)

    # Creates an upper bound for confidence
    if any(conf >= conf_bound for conf in conf_list):
        return bboxes, conf_list, phrases_np.tolist() ,True

    return bboxes, conf_list, phrases_np.tolist() ,False

# make publishing function over here for easier use

'''
def publish_seg(masks, publisher):
    # turns masks into a multiArray and then a list to be published
    segment_msg = Float32MultiArray()
    segment_msg.layout.dim.append(MultiArrayDimension())
    segment_msg.layout.dim[0].label = "segments"
    segment_msg.layout.dim[0].size = masks.numel()
    segment_msg.layout.dim[0].stride = masks.size(1)
    segment_msg.data = masks.view(-1).tolist()
    publisher.publish(segment_msg)
'''

def save_masks(masks, output_path, image_path, phrases):
    mask_arrays = masks.cpu().numpy()
    num_masks = mask_arrays.shape[0]
    height, width = mask_arrays.shape[2], mask_arrays.shape[3]

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    if image_name.endswith('_color'):
        image_name = image_name[:-6]

    image_output_path = os.path.join(output_path, image_name)
    os.makedirs(image_output_path, exist_ok=True)

    for i, mask in enumerate(mask_arrays):
        mask = mask[0]  # Assuming masks are of shape [num_masks, 1, height, width]

        # Ensure the mask is a float array for normalization
        mask = mask.astype(np.float32)
        
        # Normalize the mask to the range [0, 255]
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255.0
        mask = mask.astype(np.uint8)

        # Create a PIL Image from the array
        img = PILImg.fromarray(mask)

        phrase = phrases[i]
        prompt_output_path = os.path.join(image_output_path, phrase)
        os.makedirs(prompt_output_path, exist_ok=True)

        # Save the image
        mask_image_path = os.path.join(prompt_output_path, f"mask_{phrase}_{i}.png")
        img.save(mask_image_path)
        # print(f"Mask image saved to {mask_image_path}")

        # Display the image
        # print("Displaying mask")
        # img.show()    

def main(root_dir): 
    input_path = os.path.join(root_dir, "color")
    output_path = os.path.join(root_dir, "segments")

    # rospy.init_node('image_processor', anonymous=True)
    # bbox_pub = rospy.Publisher('bounding_boxes', Float32MultiArray, queue_size=10)
    # segment_pub = rospy.Publisher('segments', Float32MultiArray, queue_size=10)

    images = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    text_prompt = 'table'

    try:
        logging.info("Initialize object detectors")
        gdino = GroundingDINOObjectPredictor()
        SAM = SegmentAnythingPredictor()

        for image_path in images:
            # if rospy.is_shutdown():
            #     break

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


            if image_pil_bboxes[0, 3].item() >= 400:
                continue

            logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
            bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)
            print("SUCCESSFUL ITERATION")
            # bbox_annotated_pil.show()

            segments_color_path = os.path.join(root_dir, "segments_color")
            os.makedirs(segments_color_path, exist_ok=True)

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            annotated_image_path = os.path.join(segments_color_path, f"{image_name}.png")
            bbox_annotated_pil.save(annotated_image_path)

            # mask_image = PILImg.new('RGBA', image_pil.size, color=(0, 0, 0, 0))
            # mask_draw = ImageDraw.Draw(mask_image)
            # print(masks.shape)
            # for mask in masks:
            #     draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
            # mask_image.show()

            # Prepare and publish segments
            # publish_seg(masks,segment_pub)

            save_masks(masks, output_path, image_path, phrases)

            # time.sleep(15)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main(root_dir)
