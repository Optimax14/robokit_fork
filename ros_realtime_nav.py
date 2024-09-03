#!/usr/bin/env python

"""Test GroundingSAM on ros images"""

import message_filters
import cv2
import threading
import numpy as np
import rospy
from PIL import Image as PILImg

import ros_numpy

from matplotlib import pyplot as plt
from sensor_msgs.msg import Image, CameraInfo
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes

lock = threading.Lock()
from listener import ImageListener
import time
from utils import (
    compute_xyz,
    pose_in_map_frame,
    is_nearby_in_map,
    read_and_visualize_graph,
    read_graph_json,
    save_graph_json,
    denormalize_depth_image
)


class robokitRealtime:

    def __init__(self):
        # initialize a node
        rospy.init_node("seg_rgb")

        self.listener = ImageListener(camera="Fetch")

        self.counter = 0
        self.output_dir = "output/real_world"

        # initialize network
        self.text_prompt = "table, door, chair"
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()

        self.label_pub = rospy.Publisher("seg_label_refined", Image, queue_size=10)
        self.score_pub = rospy.Publisher("seg_score", Image, queue_size=10)
        self.image_pub = rospy.Publisher("seg_image", Image, queue_size=10)
        self.read_semantic_data()
        
        time.sleep(5)

    def read_semantic_data(self):
        self.graph = read_graph_json()
        self.semantic_poses = {'door':[], 'chair':[], 'table':[]}
        for node, data in self.graph.nodes(data=True):
            self.semantic_poses[data["category"]].append(data["pose"])

    def run_network(self):
        with lock:
            if self.listener.im is None:
                return
            im_color = self.listener.im.copy()
            depth_img = self.listener.depth.copy()
            rgb_frame_id = self.listener.rgb_frame_id
            rgb_frame_stamp = self.listener.rgb_frame_stamp
            RT_camera, RT_base = self.listener.RT_camera, self.listener.RT_base
        # depth_img = denormalize_depth_image(depth_image=depth_img, max_depth=20)
        print("===========================================")

        # bgr image
        im = im_color.astype(np.uint8)[:, :, (2, 1, 0)]
        img_pil = PILImg.fromarray(im)
        bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, self.text_prompt)
        if len(phrases) == 0:
            print(f"skipping zero phrases \n")
            return 
        # Scale bounding boxes to match the original image size
        w = im.shape[1]
        h = im.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        # logging.info("SAM prediction")
        image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)

        # filter large boxes
        print(masks.shape)
        image_pil_bboxes, index = filter_large_boxes(
            image_pil_bboxes, w, h, threshold=0.5
        )
        masks = masks[index]

        ##############################################################

        mask_array = masks.cpu().numpy()
        for i, mask in enumerate(mask_array):
            mask = mask[0]
            pose = pose_in_map_frame(RT_camera, RT_base, depth_img, segment=mask)
            print(f"pose {pose} class {phrases[i]}")
            if pose is None:
                continue
            self.semantic_poses[phrases[i]], is_nearby = is_nearby_in_map(self.semantic_poses[phrases[i]], pose)
            if not is_nearby:
                print(f"adding node")
                self.graph.add_node(
                    f"{phrases[i]}_new_{i}",
                    id="{phrases[i]}_new_{i}",
                    pose = pose,
                    robot_pose = None,
                    category = phrases[i]
                )
            
        ##############################################################

        mask = combine_masks(masks[:, 0, :, :]).cpu().numpy()
        gdino_conf = gdino_conf[index]
        ind = np.where(index)[0]
        phrases = [phrases[i] for i in ind]

        # logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        bbox_annotated_pil = annotate(
            overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases
        )
        # bbox_annotated_pil.show()
        im_label = np.array(bbox_annotated_pil)

        # show result
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 2, 1)
        # plt.imshow(im_label)
        # ax.set_title('output image')
        # ax = fig.add_subplot(1, 2, 2)
        # plt.imshow(mask)
        # ax.set_title('mask')
        # plt.show()

        # publish segmentation mask
        label = mask
        label_msg = ros_numpy.msgify(Image, label.astype(np.uint8), "mono8")
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = "mono8"
        self.label_pub.publish(label_msg)

        # publish score map
        score = label.copy()
        mask_ids = np.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        for index, mask_id in enumerate(mask_ids):
            score[label == mask_id] = gdino_conf[index]
        label_msg = ros_numpy.msgify(Image, score.astype(np.uint8), "mono8")
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = "mono8"
        self.score_pub.publish(label_msg)

        num_object = len(np.unique(label)) - 1
        print("%d objects" % (num_object))

        # publish segmentation images
        rgb_msg = ros_numpy.msgify(Image, im_label, "rgb8")
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)

if __name__ == "__main__":
    # image listener
    robokit_instance = robokitRealtime()
    while not rospy.is_shutdown():
        robokit_instance.run_network()
    print(f"closing script! saving graph")
    save_graph_json(robokit_instance.graph, file="graph_updated.json")
    read_and_visualize_graph("map.png","map.yaml", on_map=True, catgeories=['door', 'chair','table'], graph=robokit_instance.graph)
