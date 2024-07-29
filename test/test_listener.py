#!/usr/bin/env python

# need to use roscore or a ros environment
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # Optional, for display

def callback(data):
    img_data = np.array(data.data, dtype=np.float32)
    print(img_data.size)
    print(img_data.shape)

    num_masks = int(img_data.size / (480*640))
    masks = np.split(img_data, num_masks)

    height = 480  
    width = 640   
    
    for mask in masks:
        mask = mask.reshape((480, 640))
        
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255.0
        mask = mask.astype(np.uint8)

        img = Image.fromarray(mask)

        print("Displaying mask")
        img.show()
    
def listener():
    rospy.init_node('segments_listener', anonymous=True)
    rospy.Subscriber('segments', Float32MultiArray, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
