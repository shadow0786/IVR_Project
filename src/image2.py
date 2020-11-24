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
        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        self.yellow_img1_pos = np.array([0,0])
        self.blue_img1_pos = np.array([0,0])
        self.green_img1_pos = np.array([0,0])
        self.red_img1_pos = np.array([0,0])
        self.targetS_img1_pos = np.array([0,0])
        self.targetR_img1_pos = np.array([0,0])
        self.img1_object_data = rospy.Subscriber("/cv_image1/objects", Float64MultiArray, self.callbackIMG1)
        self.target_pos_x = rospy.Publisher("/target_pos_x",Float64,queue_size = 10)
        self.target_pos_y = rospy.Publisher("/target_pos_y",Float64,queue_size = 10)
        self.target_pos_z = rospy.Publisher("/target_pos_z",Float64,queue_size = 10)
        self.robot_Sinjoint1 = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size = 10)
        self.robot_Sinjoint2 = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size = 10)
        self.robot_Sinjoint3 = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size = 10)
        self.robot_Sinjoint4 = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size = 10)
        self.robot_joint2_angle = rospy.Publisher("/robot/joint2_angle", Float64, queue_size = 10)
        self.robot_joint3_angle = rospy.Publisher("/robot/joint3_angle", Float64, queue_size = 10)
        self.robot_joint4_angle = rospy.Publisher("/robot/joint4_angle", Float64, queue_size = 10)
        ################ CONTROL PART ######### 
  ############### QUESTION 3.1
  # initliate a publisher for the position of end effector estimated using kinematics as well as vision
        self.end_effector_estimated = rospy.Publisher("/end_effector_prediction",Float64MultiArray, queue_size=10)
        self.end_effector_kinematics = rospy.Publisher("/end_effector_kinematics",Float64MultiArray, queue_size=10)
        
  ############# QUESTION 3.2
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
    # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0,0.0,0.0], dtype='float64')
        self.error_d = np.array([0.0,0.0,0.0], dtype='float64')
        
    # initliate a publisher for the position of end effector
        self.end_effector_x = rospy.Publisher("/end_effector_x",Float64, queue_size=10)
        self.end_effector_y = rospy.Publisher("/end_effector_y",Float64, queue_size=10)
        self.end_effector_z = rospy.Publisher("/end_effector_z",Float64, queue_size=10)
    	
    	
    	
    	#### Question 4.2
        self.joints_previous = np.array([0.0000001, 0.0,0.0,0.0], dtype='float64')
        self.q_d_previous = np.array([0.0, 0.0,0.0,0.0], dtype='float64')
        self.W_past = np.array([0.0], dtype='float64')     ##Changed size 1

        self.time = rospy.get_time()
        
    def callbackIMG1(self,data):
        objects_coordinates = np.array(data.data)
        self.yellow_img1_pos =  np.array([objects_coordinates[0],objects_coordinates[1]])
        self.blue_img1_pos = np.array([objects_coordinates[2],objects_coordinates[3]])
        self.green_img1_pos = np.array([objects_coordinates[4],objects_coordinates[5]])
        self.red_img1_pos = np.array([objects_coordinates[6],objects_coordinates[7]])
        self.targetS_img1_pos = np.array([objects_coordinates[8],objects_coordinates[9]])
        self.targetR_img1_pos = np.array([objects_coordinates[10],objects_coordinates[11]]) 


    # Recieve data, process it, and publish
    def callback2(self,data):
        # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.yellow_pos_IMG2 = self.find_yellow(self.cv_image2)
        self.blue_pos_IMG2 = self.find_blue(self.cv_image2)
        self.green_pos_IMG2 = self.find_green(self.cv_image2)
        self.red_pos_IMG2 = self.find_red(self.cv_image2)
        templateS = "/home/amami/Project/catkin_ws/src/ivr_assignment/src/image_crop.png" ## CHANGE
        ### change
        templateR = "/home/amami/Project/catkin_ws/src/ivr_assignment/src/image_box1.png"
        
        
        self.orange_target_ = self.target_3D_location(self.cv_image2, templateS) ## DEBUG
        self.orange_box = self.target_3D_location1(self.cv_image2, templateR) # change
        
        self.combined_posIMG12 = self.pos_3D_plane()
        
        self.Sinjoint2=Float64()
        self.Sinjoint2.data= self.getSinJoints()[0]
        self.Sinjoint3=Float64()
        self.Sinjoint3.data= self.getSinJoints()[1]
        self.Sinjoint4=Float64()
        self.Sinjoint4.data= self.getSinJoints()[2]
        self.joint2=Float64()       
        self.joint2.data= self.get_joint_angles(self.pos_3D_plane())[0]
        
        self.joint3=Float64()
        self.joint3.data= self.get_joint_angles(self.pos_3D_plane())[1]
        self.joint4=Float64()
        self.joint4.data= self.get_joint_angles(self.pos_3D_plane())[2] 
        self.target = Float64()
        self.target= self.orange_target_  
        self.target_x = Float64() 
        self.target_y = Float64()
        self.target_z = Float64()
        self.target_x= self.orange_target_[0]
        self.target_y= self.orange_target_[1] 
        self.target_z= self.orange_target_[2]
       
        
        ###### CONTOL PART QUESTION 3.1
        cal_joints = self.get_joint_angles(self.pos_3D_plane())
        joints = np.array([0,cal_joints[0],cal_joints[1],cal_joints[2]])
        x_e =  self.forward_kinematics(joints)
        self.end_effector = Float64MultiArray()
        self.end_effector.data = self.combined_posIMG12 [3]
        self.end_effector_kin = Float64MultiArray()
        self.end_effector_kin.data = x_e
        
        ### QUESTION 3.2
        q_d = self.control_closed()
        end_eff = self.pos_3D_plane()[3]
        self.joint1_control = Float64()
        self.joint1_control.data = q_d[0]
        self.joint2_control = Float64()
        self.joint2_control.data = q_d[1]
        self.joint3_control = Float64()
        self.joint3_control.data = q_d[2]
        self.joint4_control = Float64()
        self.joint4_control.data = q_d[3]
        
        self.end_eff_x = Float64()
        self.end_eff_x.data = end_eff[0]
        self.end_eff_y = Float64()
        self.end_eff_y.data = end_eff[1]
        self.end_eff_z = Float64()
        self.end_eff_z.data = end_eff[2]


        # Uncomment if you want to save the image
        #cv2.imwrite('image_copy.png', cv_image)
        #im2=cv2.imshow('window2', self.cv_image2)
        #cv2.waitKey(1)

        # Publish the results
        try: 
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
            ### Uncomment for vision part
            #self.robot_Sinjoint2.publish(self.Sinjoint2)
            #self.robot_Sinjoint3.publish(self.Sinjoint3)
            #self.robot_Sinjoint4.publish(self.Sinjoint4)
            ### Uncomment for QUESTION 3.2
            #self.robot_Sinjoint1.publish(self.joint1_control) ## Changed delete it !
            self.robot_Sinjoint2.publish(self.joint2_control)
            self.robot_Sinjoint3.publish(self.joint3_control)
            self.robot_Sinjoint4.publish(self.joint4_control)
            
            ## 
            self.robot_joint2_angle.publish(self.joint2)
            self.robot_joint3_angle.publish(self.joint3)
            self.robot_joint4_angle.publish(self.joint4)
            self.target_pos_x.publish(self.target_x)
            self.target_pos_y.publish(self.target_y)
            self.target_pos_z.publish(self.target_z)
     
     #### Question 3.1       
            self.end_effector_estimated.publish(self.end_effector) 
            self.end_effector_kinematics.publish(self.end_effector_kin)
            self.end_effector_x.publish(self.end_eff_x)
            self.end_effector_y.publish(self.end_eff_y)
            self.end_effector_z.publish(self.end_eff_z)
            

        except CvBridgeError as e:
            print(e)
        
    def pos_3D_plane(self):

        meter = self.pixTometer() 
        yellow_pos_3D = np.array([(self.yellow_pos_IMG2[0] - self.yellow_pos_IMG2[0]) * meter,(self.yellow_img1_pos[0] -self.yellow_img1_pos[0]) *meter,((self.yellow_img1_pos[1]  - self.yellow_img1_pos[1]) *-1) *meter])
        blue_pos_3D = np.array([(self.blue_pos_IMG2[0] - self.yellow_pos_IMG2[0]) * meter,(self.blue_img1_pos[0] -self.yellow_img1_pos[0]) *meter,((self.blue_img1_pos[1]  - self.yellow_img1_pos[1]) *-1) *meter])
        green_pos_3D = np.array([(self.green_pos_IMG2[0] - self.yellow_pos_IMG2[0]) * meter,(self.green_img1_pos[0] -self.yellow_img1_pos[0]) *meter,((self.green_img1_pos[1]  - self.yellow_img1_pos[1]) *-1) *meter])
        red_pos_3D = np.array([(self.red_pos_IMG2[0] -self.yellow_pos_IMG2[0]) * meter,(self.red_img1_pos[0] -self.yellow_img1_pos[0]) *meter,((self.red_img1_pos[1]  - self.yellow_img1_pos[1]) *-1) *meter])
        
        if (self.green_pos_IMG2[0] == -10):
            if (self.blue_pos_IMG2[0] == -10):
                green_pos_3D = np.array([0, (self.green_img1_pos[0] - self.yellow_img1_pos[0])  * meter, ((self.green_img1_pos[1] - self.yellow_img1_pos[1])*-1) * meter])
            if (self.blue_pos_IMG2[0] != -10):
                if (self.red_pos_IMG2[0] == -10):
                    green_pos_3D = np.array([0, (self.green_img1_pos[0] - self.yellow_img1_pos[0])  * meter, ((self.green_img1_pos[1] - self.yellow_img1_pos[1])*-1) * meter])
                if (self.red_pos_IMG2[0] != -10):
                    if (self.red_img1_pos[1] - self.yellow_img1_pos[1] > 0):
                        green_pos_3D = np.array([0, (self.green_img1_pos[0] - self.yellow_img1_pos[0])  * meter, ((self.green_img1_pos[1] - self.yellow_img1_pos[1])*-1) * meter])
                    if (self.red_img1_pos[1] - self.yellow_img1_pos[1] < 0):
                        green_pos_3D = np.array([0 , (self.red_img1_pos[0] - self.yellow_img1_pos[0]) *meter , ((self.red_img1_pos[1] - self.yellow_img1_pos[1])*-1) * meter])
        
        if (self.red_pos_IMG2[0] == -10):  
            red_pos_3D = np.array([(self.green_img1_pos[0] - self.yellow_img1_pos[0]) * meter, (self.red_img1_pos[0] - self.yellow_img1_pos[0]) *meter , ((self.red_img1_pos[1] - self.yellow_img1_pos[1])*-1) * meter])
        
        if (self.green_img1_pos[0] == -10):
            green_pos_3D = np.array([(self.green_pos_IMG2[0] - self.yellow_pos_IMG2[0]) * meter, 0 , ((self.green_pos_IMG2[1] - self.yellow_pos_IMG2[1])*-1) * meter])
        if (self.red_img1_pos[0] == -10):
            red_pos_3D = np.array([(self.red_pos_IMG2[0] - self.yellow_pos_IMG2[0]) * meter, 0 , ((self.red_pos_IMG2[1] - self.yellow_pos_IMG2[1])*-1) * meter])
        
        return np.array([yellow_pos_3D , blue_pos_3D , green_pos_3D , red_pos_3D])
    
    def rotation_matrix_y(self, angle):
        R_y = np.array([[np.cos(angle),0,-np.sin(angle)],
                            [0,1,0],
                            [np.sin(angle), 0, np.cos(angle)]])
        return R_y

    def rotation_matrix_x(self, angle):
        R_x = np.array([ [1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]])
        return R_x
    
    def get_joint_angles(self, pos_3D_plane):
        [yellow, blue, green, red] = pos_3D_plane
        link2 = green - blue

        ### start with joint 2 since joint 1 does not rotate
        angle2 = np.arctan2(green[2] - blue[2], green[1] - blue[1])     

        ####### transform the coordinates into the rotated space
        rotation_matrix_2 = self.rotation_matrix_x(-angle2 + np.pi/2)
        yellow2 = np.dot(rotation_matrix_2,yellow)
        blue2 = np.dot(rotation_matrix_2, blue)
        green2 = np.dot(rotation_matrix_2, green)
        red2 = np.dot(rotation_matrix_2, red)

        ########## calculate joint angle 3 in the new rotated space
        angle3 = np.arctan2(green2[2] - blue2[2], green2[0] - blue2[0])  - np.pi/2   
        
        angle3_temp = np.arctan2(green[2] - blue[2], green[0] - blue[0])
        rotation_matrix = self.rotation_matrix_y(-angle3_temp + np.pi/2)
        yellow1 = np.dot(rotation_matrix,yellow)
        blue1 = np.dot(rotation_matrix, blue)
        green1 = np.dot(rotation_matrix, green)
        red1 = np.dot(rotation_matrix, red)
        angle2 = np.arctan2(green1[2] - blue1[2], green1[1] - blue1[1]) - np.pi/4

        ####### transform the coordinates into the rotated space
        rotation_matrix_3 = self.rotation_matrix_y(-angle3)
        yellow3 = np.dot(rotation_matrix_3,yellow)
        blue3 = np.dot(rotation_matrix_3, blue)
        green3 = np.dot(rotation_matrix_3, green)
        red3 = np.dot(rotation_matrix_3, red)
        
        rotation_matrix_4 = self.rotation_matrix_x(-angle2)  # 
        yellow4 = np.dot(rotation_matrix_4,yellow3)
        blue4 = np.dot(rotation_matrix_4, blue3)
        green4 = np.dot(rotation_matrix_4, green3)
        red4 = np.dot(rotation_matrix_4, red3)

        ########## calculate joint angle 3 in the new rotated space
        angle4 = np.arctan2(red4[2] - green4[2], red4[1] - green4[1])    


        return [angle2, -angle3 , angle4 -0.6]  ##CHANGED
        # Solve using trigonometry
    # 
        
    def getSinJoints(self):
        cur_time = np.array([rospy.get_time()]) - self.time
        self.joint2 = float(np.pi/2 * np.sin(np.pi/15 * cur_time)) 
        self.joint3 = float(np.pi/2 * np.sin(np.pi/18 * cur_time))
        self.joint4 = float(np.pi/2 * np.sin(np.pi/20 * cur_time))
        return np.array([self.joint2, self.joint3,self.joint4])  
    
    
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
    
    def target_3D_location(self , image1 , template):

        cam2 = self.locate_target_sphere(image1, template)

        sphere_location = np.array([cam2[0] - self.yellow_pos_IMG2[0], self.targetS_img1_pos[0] - self.yellow_img1_pos[0], ( cam2[1] - self.yellow_pos_IMG2[1])*-1]) * self.pixTometer()
    
        if (cam2[1] == -10):
            target_z = self.targetS_img1_pos[1]
            sphere_location = np.array([cam2[0] - self.yellow_pos_IMG2[0], self.targetS_img1_pos[0] - self.yellow_img1_pos[0], (target_z  - self.yellow_img1_pos[1])*-1]) * self.pixTometer()
           
        if (self.targetS_img1_pos[1] == -10):
            target_z =cam2[1]
            sphere_location = np.array([cam2[0] - self.yellow_pos_IMG2[0], self.targetS_img1_pos[0] - self.yellow_img1_pos[0], (target_z  - self.yellow_pos_IMG2[1])*-1]) * self.pixTometer()
        
        sphere_location = np.array([sphere_location[0]+0.6,sphere_location[1]+0.6,sphere_location[2] - 0.7])  ## CHANGED
        return sphere_location
        
        ######## DEBUG
        
    def target_3D_location1(self , image1 , template):

        cam2 = self.locate_target_sphere(image1, template)

        sphere_location = np.array([cam2[0] - self.yellow_pos_IMG2[0], self.targetR_img1_pos[0] - self.yellow_img1_pos[0], ( cam2[1] - self.yellow_pos_IMG2[1])*-1]) * self.pixTometer()
    
        if (cam2[1] == -10):
            target_z = self.targetR_img1_pos[1]
            sphere_location = np.array([cam2[0] - self.yellow_pos_IMG2[0], self.targetR_img1_pos[0] - self.yellow_img1_pos[0], (target_z  - self.yellow_img1_pos[1])*-1]) * self.pixTometer()
           
        if (self.targetR_img1_pos[1] == -10):
            target_z =cam2[1]
            sphere_location = np.array([cam2[0] - self.yellow_pos_IMG2[0], self.targetR_img1_pos[0] - self.yellow_img1_pos[0], (target_z  - self.yellow_pos_IMG2[1])*-1]) * self.pixTometer()
        
        sphere_location = np.array([sphere_location[0]+0.6,sphere_location[1]+0.6,sphere_location[2] - 0.7])  ## CHANGED
        return sphere_location  
        
        
    

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
        NewMask = cv2.dilate(mask , kernel , iterations = 3)
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
    
    def pixTometer(self):
        distance = 2.5/(self.blue_pos_IMG2[1] - self.yellow_pos_IMG2[1])
        return np.abs(distance)
        
     #################### CONTROL PART QUESTION 3.1
    def forward_kinematics(self,joints):
      s1 = np.sin(joints[0])
      s2 = np.sin(joints[1])
      s3 = np.sin(joints[2])
      s4 = np.sin(joints[3])
      c1 = np.cos(joints[0])
      c2 = np.cos(joints[1])
      c3 = np.cos(joints[2])
      c4 = np.cos(joints[3])
      l1 = 2.5
      l3 = 3.5
      l4 = 3.0
      end_effector = np.array([(s1 * s2 * c3 + c1 * s3) * (l3 + l4 * c4) + (l4 * s1 * c2 * s4),
                             (s1 * s3 - c1 * s2 * c3) * (l3 + l4 * c4) - (l4 * c1 * c2 * s4),
                             c2 * c3 * (l3 + l4 * c4) - (l4 * s2 * s4) + l1])
      return end_effector
      
      ######### QUESTION 3.2
    def jacobian_matrix(self, joints):
      s1 = np.sin(joints[0])
      s2 = np.sin(joints[1])
      s3 = np.sin(joints[2])
      s4 = np.sin(joints[3])
      c1 = np.cos(joints[0])
      c2 = np.cos(joints[1])
      c3 = np.cos(joints[2])
      c4 = np.cos(joints[3])
      l1 = 2.5
      l3 = 3.5
      l4 = 3.0
      jacobian = np.array(
      [[l4 * c1 * s2 * c3 * c4 - l4 * s1 * s3 * c4 + l4 * c1 * c2 * s4 + l3 * c1 * s2 * c3 - l3 * s1 * s3,
        l4 * s1 * c2 * c3 * c4 - l4 * s1 * s2 * s4 + l3 * s1 * c2 * c3,
        -l4 * s1 * s2 * s3 * c4 + l4 * c1 * c3 * c4 - l3 * s1 * s2 * s3 + l3 * c1 * c3,
        -l4 * s1 * s2 * c3 * s4 - l4 * c1 * s3 * s4 + l4 * s1 * c2 * c4],
       [l4 * s1 * s2 * c3 * c4 + l4 * c1 * s3 * c4 + l4 * s1 * c2 * s4 + l3 * s1 * s2 * c3 + l3 * c1 * s3,
        -l4 * c1 * c2 * c3 * c4 + l4 * c1 * s2 * s4 - l3 * c1 * c2 * c3,
        l4 * c1 * s2 * s3 * c4 + l4 * s1 * c3 * c4 + l3 * c1 * s2 * s3 + l3 * s1 * c3,
        l4 * c1 * s2 * c3 * s4 - l4 * s1 * s3 * s4 - l4 * c1 * c2 * c4],
       [0, -l4 * s2 * c3 * c4 - l4 * c2 * s4 - l3 * s2 * c3, -l4 * c2 * s3 * c4 - l3 * c2 * s3,
        -l4 * c2 * c3 * s4 - l4 * s2 * c4]])
      return jacobian
      
    def control_closed(self):
      # P gain
      K_p = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
      # D gain
      K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
      # estimate time step
      cur_time = np.array([rospy.get_time()])
      dt = cur_time - self.time_previous_step
      self.time_previous_step = cur_time
      # robot end-effector position
      pos = self.combined_posIMG12 [3] 
      # desired trajectory
      pos_d = self.orange_target_  ### target Position ball
      #pos_square = self.box_coor  ### target Position box USED 4.2
      pos_square = self.orange_box


      
      # estimate derivative of error
      self.error_d = ((pos_d - pos) - self.error) / dt
      # estimate error
      self.error = pos_d - pos
      # estimate current joint angles using vision
      joints = self.get_joint_angles(self.pos_3D_plane())
      q = np.array([ 0 ,joints[0],joints[1],joints[2] ]) ## Problem ! it should not be 0 ! #### CHANGE : IT should it is not updated
      #q = self.q_d_previous
      
      #### END CHange
      #J_inv = np.linalg.pinv(self.calculate_jacobian(q))  # calculating the psudeo inverse of Jacobian -- wrong name CHANGED
      J_inv = np.linalg.pinv(self.jacobian_matrix(q))
      dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p,                                                   self.error.transpose())))  # control input (angular velocity of joints)
      
    ############## USED in 4.2
      W = np.sum((pos - pos_square) ** 2)
      q0 = (W - self.W_past) / (q - self.joints_previous)  ## is q being updated ? otherwise we might get 0 all the time !
      self.joints_previous = q
      self.joints_previous[0] = 0.0000001
      self.W_past = W  
      
      I = np.array([[1, 0, 0,0], [0, 1,0, 0], [0, 0, 1,0],[0, 0, 0,1]])
       #dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose()))) ###already done
      dq_d = dq_d + (np.dot(I - np.dot(J_inv, self.jacobian_matrix(q)), q0))
       
      
        
    ###############
      q_d = q + (dt * dq_d)  # control input (angular position of joints)
      #self.q_d_previous = q_d  ########## NEW : Used to derive derivitive Part 4.2 ;
      return q_d  

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
