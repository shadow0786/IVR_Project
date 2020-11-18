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

    ####CONTROL PART##
    # record the begining time
    #self.time_trajectory = rospy.get_time()
    # initialize a publisher to send robot end-effector position
    self.end_effector_pub = rospy.Publisher("end_effector_prediction",Float64MultiArray, queue_size=10)
    # initialize a publisher to send desired trajectory
    self.trajectory_pub = rospy.Publisher("trajectory",Float64MultiArray, queue_size=10)
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size = 10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size = 10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size = 10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size = 10)
    # initialize errors
    self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
    # initialize error and derivative of error for trajectory tracking
    self.error = np.array([0.0,0.0,0.0], dtype='float64')
    self.error_d = np.array([0.0,0.0,0.0], dtype='float64')
    #self.pos_previous = np.array([0.0, 0.0,0.0], dtype='float64')
    #self.W_past = np.array([0.0, 0.0,0.0], dtype='float64')

  ##################################### Vision part
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
    angle2 = atan2(green[2] - blue[2], green[1] - blue[1])     ### should we subtract pi/2  ?

    ####### transform the coordinates into the rotated space
    rotation_matrix_2 = self.rotation_matrix_x(-angle2)
    yellow2 = np.dot(rotation_matrix_2,yellow)
    blue2 = np.dot(rotation_matrix_2, blue)
    green2 = np.dot(rotation_matrix_2, green)
    red2 = np.dot(rotation_matrix_2, red)

    ########## calculate joint angle 3 in the new rotated space
    angle3 = atan2(green2[2] - blue2[2], green2[0] - blue2[0])     ### should we subtract pi/2 ?

    ####### transform the coordinates into the rotated space
    rotation_matrix_3 = self.rotation_matrix_y(-angle3)
    yellow3 = np.dot(rotation_matrix_2,yellow2)
    blue3 = np.dot(rotation_matrix_2, blue2)
    green3 = np.dot(rotation_matrix_2, green2)
    red3 = np.dot(rotation_matrix_2, red2)

    ########## calculate joint angle 3 in the new rotated space
    angle4 = atan2(green3[2] - blue3[2], green3[1] - blue3[1])     ### should we subtract pi/2


    return [angle2, angle3 , angle4]






  ############################## THIS IS QUESTION 3.1
  # As a general assumption : joints position and orange sphere/square position are already calculated in functions named
  # Array of 4 joints = detect_joint_angles(image)
  # trajectory() = sphere position [x,y,z]
  # trajectory1() = square position [x,y,z]
  def detect_end_effector(self, image):
    a = self.pixel2meter(image)
    endPos = a * (self.detect_yellow(image) - self.detect_red(image))
    return endPos

  # Calculate the forward kinematics
  def forward_kinematics(self, image):
    joints = self.detect_joint_angles(image)
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

  return end_effector  # if results are not correct : Consider changing the matrix by substraction position of red circle from corresponding row

  ######################################   THIS IS QUESTION 3.2
  def jacobian_matrix(self, q):
    joints = self.detect_joint_angles(image)
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

  def control_closed(self, image):
    # P gain
    K_p = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    # D gain
    K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    # estimate time step
    cur_time = np.array([rospy.get_time()])
    dt = cur_time - self.time_previous_step
    self.time_previous_step = cur_time
    # robot end-effector position
    pos = self.detect_end_effector(image)
    # desired trajectory
    pos_d = self.trajectory()  ### target Position ball
    pos_square = self.trajectory1()  ### target Position square

    ############################ Used in 4.2
    # W = np.sum((pos - self.pos_square) ** 2)
    # q0 = (W - W_past) / (pos - pos_previous)
    # self.pos_previous = pos  ########## NEW : Used to derive derivitive Part 4.2 ; Needs to be initialized ? should be defined as global variable ? self.pos ?
    # self.W_past = W  #### W need to be initialized ? should be defined as global variable ? self.W ?
    ############################
    # estimate derivative of error
    self.error_d = ((pos_d - pos) - self.error) / dt
    # estimate error
    self.error = pos_d - pos
    q = self.detect_joint_angles(image)  # estimate initial value of joints'
    J_inv = np.linalg.pinv(self.calculate_jacobian(q))  # calculating the psudeo inverse of Jacobian
    dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p,
                                                                         self.error.transpose())))  # control input (angular velocity of joints)
    ############## USED in 4.2
    # I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose())))
    # dq_d = dq_d + (np.dot(I - np.dot(J_inv, self.calculate_jacobian(q)), q0))
    ###############
    q_d = q + (dt * dq_d)  # control input (angular position of joints)
    return q_d

  # Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    ################################ QUESTION 3.1
    # compare the estimated position of robot end-effector calculated from images with forward kinematics(10 Values in a Table)
    x_e = self.forward_kinematics(cv_image)
    x_e_image = self.detect_end_effector(cv_image)   #### this is not being published ... Check Try : (needs more discussion with teammate)
    self.end_effector = Float64MultiArray()
    self.end_effector.data = x_e_image
    #################################### Question 3.2 and 4.2
    # send control commands to joints (lab 3)
    q_d = self.control_closed(cv_image)
    self.joint1 = Float64()
    self.joint1.data = q_d[0]
    self.joint2 = Float64()
    self.joint2.data = q_d[1]
    self.joint3 = Float64()
    self.joint3.data = q_d[2]
    self.joint4 = Float64()
    self.joint4.data = q_d[3]


    # Publishing the desired trajectory on a topic named trajectory  ### Trajectory of the orange Sphere : To be compared with the end_effector position (plot)
    x_d = self.trajectory()  # getting the desired trajectory
    self.trajectory_desired = Float64MultiArray()
    self.trajectory_desired.data = x_d


    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      ########### Publish results for Question 3
      self.end_effector_pub.publish(self.end_effector)
      self.trajectory_pub.publish(self.trajectory_desired)
      self.robot_joint1_pub.publish(self.joint1)
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
      self.robot_joint4_pub.publish(self.joint4)
    except CvBridgeError as e:
      print(e)

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


