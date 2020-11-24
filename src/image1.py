#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:
  # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        self.cv_image1_data_pub = rospy.Publisher("/cv_image1/objects" , Float64MultiArray, queue_size = 12) # mistake size not length


    # Recieve data from camera 1, process it, and publish
    def callback1(self,data):
        # Recieve the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.objects = Float64MultiArray()
        green = self.find_green(self.cv_image1)
        yellow = self.find_yellow(self.cv_image1)
        red = self.find_red(self.cv_image1)
        blue = self.find_blue(self.cv_image1)
        templateS = "/home/amami/Project/catkin_ws/src/ivr_assignment/src/image_crop.png" # change
        templateR = "/home/amami/Project/catkin_ws/src/ivr_assignment/src/image_box1.png" # change
        target_R = self.locate_target_sphere(self.cv_image1, templateR) # change

        
        target_S = self.locate_target_sphere(self.cv_image1, templateS)
    #     target_R = self.locate_targets(self, cv_image1, templateS , templateR)[1]
        
        data = self.makeData(yellow , blue , green , red , target_S,target_R) # change
        self.objects.data = data
        
    
        # Uncomment if you want to save the image
        #cv2.imwrite('image_copy.png', cv_image)

        #im1=cv2.imshow('window1', self.cv_image1)
        #cv2.waitKey(1)
        # Publish the results
        try: 
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
            self.cv_image1_data_pub.publish(self.objects)
        except CvBridgeError as e:
            print(e)

    def makeData(self , array , array1 , array2 , array3 , array4,array5):
        return np.array([array[0] , array[1] , array1[0] , array1[1] , array2[0] , array2[1] , array3[0] , array3[1] , array4[0] , array4[1],array5[0],array5[1]]) # change
        
    def locate_target_sphere(self, image1, templateS):
        maskOrange = self.find_orange(image1)
        templateSphere = cv2.imread(templateS, 0)
        matching = cv2.matchTemplate(maskOrange, templateSphere, 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching)
        if (min_loc[1] == 0):
            return np.array([-10,-10])
        x = min_loc[0] 
        y = min_loc[1]
        return np.array([x,y])
    
        
    def find_yellow(self , image):

        mask = cv2.inRange(image , (0,100,100) , (0,255,255))
        kernel = np.ones((5,5) , np.uint8)
        NewMask = cv2.dilate(mask , kernel , iterations = 3)
        Mo = cv2.moments(NewMask)
        if (Mo['m00'] == 0):
            cx = -10
            cy = -10
            return np.array([cx,cy])
        cx = int(Mo['m10'] / Mo['m00'])
        cy = int(Mo['m01'] / Mo['m00'])
        return np.array([cx,cy])
    
    def find_red(self , image):

        mask = cv2.inRange(image , (0,0,100) , (0,0,255))
        kernel = np.ones((5,5) , np.uint8)
        NewMask = cv2.dilate(mask , kernel , iterations = 3)
        Mo = cv2.moments(NewMask)
        if (Mo['m00'] == 0):
            cx = -10
            cy = -10
            return np.array([cx,cy])
        cx = int(Mo['m10'] / Mo['m00'])
        cy = int(Mo['m01'] / Mo['m00'])
        return np.array([cx,cy])
    
    def find_green(self , image):

        mask = cv2.inRange(image , (0,100,0) , (0,255,0))
        kernel = np.ones((5,5) , np.uint8)
        NewMask = cv2.dilate(mask , kernel , iterations = 3)
        Mo = cv2.moments(NewMask)
        if (Mo['m00'] == 0):
            cx = -10
            cy = -10
            return np.array([cx,cy])
        cx = int(Mo['m10'] / Mo['m00'])
        cy = int(Mo['m01'] / Mo['m00'])
        return np.array([cx,cy])
    
    def find_blue(self , image):

        mask = cv2.inRange(image , (100,0,0) , (255,0,0))
        kernel = np.ones((5,5) , np.uint8)
        NewMask = cv2.dilate(mask , kernel , iterations= 3)
        Mo = cv2.moments(NewMask)
        if (Mo['m00'] == 0):
            cx = -10
            cy = -10
            return np.array([cx,cy])
        cx = int(Mo['m10'] / Mo['m00'])
        cy = int(Mo['m01'] / Mo['m00'])
        return np.array([cx,cy])
    
    def find_orange(self , image):
        image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(image_hsv  ,(9,100,100) , (29,255,255))
        return mask            

    # call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:

        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
    
