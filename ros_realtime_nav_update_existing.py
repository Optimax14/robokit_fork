#!/usr/bin/env python

"""Test GroundingSAM on ros images"""

import message_filters
import cv2
import threading
import numpy as np
import rospy
from PIL import Image as PILImg
import os
import ros_numpy
import networkx as nx
from networkx import Graph

from matplotlib import pyplot as plt
from sensor_msgs.msg import Image, CameraInfo
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes, filter
from shapely.geometry import Point, Polygon

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
    denormalize_depth_image,
    get_fov_points_in_map
)
import datetime

class robokitRealtime:

    def __init__(self):
        # initialize a node
        rospy.init_node("seg_rgb")

        self.listener = ImageListener(camera="Fetch")

        self.counter = 0
        self.output_dir = "output/real_world"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.segments_dir = os.path.join(os.getcwd(), current_time)
        os.makedirs(self.segments_dir)
        # initialize network
        self.text_prompt = "table . door . chair ."
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()
        self.threshold = {"table": 1.3, "chair":0.5
                          , "door": 1}

        # self.label_pub = rospy.Publisher("seg_label_refined", Image, queue_size=10)
        # self.score_pub = rospy.Publisher("seg_score", Image, queue_size=10)
        self.image_pub = rospy.Publisher("seg_image", Image, queue_size=10)
        # self.read_semantic_data()
        self.graph = read_graph_json("graph.json")
        self.pose_list = {"table":[], "chair":[], "door":[]}
        time.sleep(5)

    # def read_semantic_data(self):
    #     self.graph = read_graph_json()
    #     self.semantic_poses = {'door':[], 'chair':[], 'table':[]}
    #     for node, data in self.graph.nodes(data=True):
    #         self.semantic_poses[data["category"]].append(data["pose"])

    def run_network(self, iter_):
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

        # bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, self.text_prompt)
        # Adding Itay's filter values
        bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, self.text_prompt,0.55, 0.55)
        bboxes, gdino_conf, phrases, flag = filter(bboxes, gdino_conf, phrases, 1, 0.8, 0.8, 0.8, 0.01, True)
        if flag:
            # print(f"flag {flag}")
            fov_points = get_fov_points_in_map(depth_img, RT_camera, RT_base)
            fov = Polygon(fov_points)
            nodes_in_fov = {}
            for node, data in self.graph.nodes(data=True):
                if "new" in node:
                    continue
                pose_  = data["pose"]
                pose_[2]=0
                # print(f"node {node}")
                # print(f"pose_ {pose_}")
                point = Point(pose_)
                
                if fov.contains(point):
                    # print(fov_points,"\n")
                    # print(f"contains -----------")
                    # print(f"node {node}")
                    # print(f"pose_ {pose_}")
                    nodes_in_fov[node] = data["category"]
            # print(f"nodes in fov {nodes_in_fov}")
            # print(fov_points,"\n")
            for nodes in nodes_in_fov.keys():
                print(f"node is being removed {nodes}")
                self.graph.remove_node(nodes)
                # print(f"node {node} removed")
            # time.sleep(10)
            return

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
        detected_poses = {"door":[],"chair":[],"table":[]}
        mask_array = masks.cpu().numpy()
        for i, mask in enumerate(mask_array):
            mask = mask[0]
            pose = pose_in_map_frame(RT_camera, RT_base, depth_img, segment=mask)
            # print(f"pose {pose} class {phrases[i]}")
            if pose is None:
                continue
            detected_poses[phrases[i]].append(pose)


        fov_points = get_fov_points_in_map(depth_img, RT_camera, RT_base)
        # print(fov_points,"\n")
        fov = Polygon(fov_points)
        nodes_in_fov = {}
        for node, data in self.graph.nodes(data=True):
            if "new" in node:
                continue
            pose_  = data["pose"]
            pose_[2]=0
            point = Point(pose_)
            
            if fov.contains(point):
                if len(detected_poses[data["category"]]) ==0:
                    nodes_in_fov[node] = data["category"]
                elif np.any(np.linalg.norm(np.array(detected_poses[data["category"]])-np.array(pose_), axis =1)) < self.threshold[data["category"]]:
                    # print(f"node present")
                    continue
                else:
                    # self.graph.remove_node(node)

                    nodes_in_fov[node] = data["category"]
        #             print(f"node removed")

        # print(f"nodes in fov {nodes_in_fov}")
        # print(fov_points,"\n")

        for nodes in nodes_in_fov.keys():
            self.graph.remove_node(nodes)
            print(f"node is being removed {nodes}")

        # print(nodes_in_fov)

        

        # for node in nodes_in_fov:



        # 1. get fov, get the nodes.
        # 2. get the detected poses. 
        # 3. for each node in fov, check if a detection exist or not nearby. if not, remove the node.
        # 4. for each detection, check if there is a nearby in graph. if not add it to the graph.

        # mask_array = masks.cpu().numpy()
        phrase_iter_ = {"table": 0, "door": 0, "chair": 0}
        for i, mask in enumerate(mask_array):
            mask = mask[0]
            pose = pose_in_map_frame(RT_camera, RT_base, depth_img, segment=mask)
            # print(f"pose {pose} class {phrases[i]}")
            if pose is None:
                continue
            # self.semantic_poses[phrases[i]], is_nearby = is_nearby_in_map(self.semantic_poses[phrases[i]], pose)
            self.pose_list[phrases[i]], _is_nearby = is_nearby_in_map(
                        self.pose_list[phrases[i]], pose, threshold=self.threshold[phrases[i]]
                    )
            if not _is_nearby:
                print(f"adding node")
                self.graph.add_node(
                    f"new_{phrases[i]}_{iter_}_{phrase_iter_[phrases[i]]}",
                    id=f"new_{phrases[i]}_{iter_}_{phrase_iter_[phrases[i]]}",
                    pose = pose,
                    robot_pose = RT_base.tolist(),
                    category = phrases[i],
                )
                phrase_iter_[phrases[i]] += 1
            self.pose_list[phrases[i]].append(pose)
            
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
        # label = mask
        # label_msg = ros_numpy.msgify(Image, label.astype(np.uint8), "mono8")
        # label_msg.header.stamp = rgb_frame_stamp
        # label_msg.header.frame_id = rgb_frame_id
        # label_msg.encoding = "mono8"
        # self.label_pub.publish(label_msg)

        # publish score map
        # score = label.copy()
        # mask_ids = np.unique(label)
        # if mask_ids[0] == 0:
        #     mask_ids = mask_ids[1:]
        # for index, mask_id in enumerate(mask_ids):
        #     score[label == mask_id] = gdino_conf[index]
        # label_msg = ros_numpy.msgify(Image, score.astype(np.uint8), "mono8")
        # label_msg.header.stamp = rgb_frame_stamp
        # label_msg.header.frame_id = rgb_frame_id
        # label_msg.encoding = "mono8"
        # self.score_pub.publish(label_msg)

        # num_object = len(np.unique(label)) - 1
        # print("%d objects" % (num_object))

        # publish segmentation images
        rgb_msg = ros_numpy.msgify(Image, im_label, "rgb8")
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)

if __name__ == "__main__":
    # image listener
    robokit_instance = robokitRealtime()
    iter_= 0
    while not rospy.is_shutdown():
        robokit_instance.run_network(iter_=iter_)
        iter_ += 1
    print(f"closing script! saving graph")
    save_graph_json(robokit_instance.graph, file="graph_updated.json")
    read_and_visualize_graph("map.png","map.yaml", on_map=True, catgeories=['door', 'chair','table'], graph=robokit_instance.graph)
