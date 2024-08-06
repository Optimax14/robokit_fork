#!/usr/bin/env python


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

    # Assuming you know the image dimensions (height, width)
    height = 480  # replace with actual height
    width = 640   # replace with actual width
    
    # Reshape the data into an image
    for mask in masks:
        mask = mask.reshape((480, 640))
        
        # Normalize the data to the range [0, 255] if necessary
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255.0
        mask = mask.astype(np.uint8)

        # Create a PIL Image from the array
        img = Image.fromarray(mask)

        # Display the image
        print("Displaying mask")
        img.show()
    
def listener():
    # Initialize the ROS node
    rospy.init_node('segments_listener', anonymous=True)
    
    # Subscribe to the 'segments' topic
    rospy.Subscriber('segments', Float32MultiArray, callback)
    
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    listener()
