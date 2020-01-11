#!/usr/bin/env python
# license removed for brevity
import rospy
import math
import tf
from std_msgs.msg import String
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
from geometry_msgs.msg import TwistStamped, PoseStamped



orientation = 0.0

def pose(data):
    global orientation
    qw = data.pose.orientation.w
    qx = data.pose.orientation.x
    qy = data.pose.orientation.y
    qz = data.pose.orientation.z
    (r,p,y) = tf.transformations.euler_from_quaternion([qx,qy,qz,qw])
    orientation = math.atan2(math.sin(y),math.cos(y))

def tracker():
    global orientation
    rospy.Subscriber("/mavros/local_position/pose",PoseStamped,pose)
    pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped, queue_size = 1)
    rospy.init_node('tracker', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    args = vars(ap.parse_args())


    lower_hsv = np.array([0,109,235])
    upper_hsv = np.array([28,202,255])

    xo = 600/2
    yo = 450/2

    x=600
    y=450

    kp = 0.5
    kd = 0.2

    x_1 = 0
    y_1 = 0

    cmd = TwistStamped()

    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])


    while not rospy.is_shutdown():
        (grabbed, frame) = camera.read()

        if args.get("video") and not grabbed:
            break

        frame = imutils.resize(frame, width=600)
        frame = cv2.medianBlur(frame,9)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        cv2.imshow("Mask", mask)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=3)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > 5:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        ex = x - xo
        ey = y - yo

        if len(cnts) > 0:
            ux = kp*ex + kd*(ex-x_1)
            uy = kp*ey + kd*(ey-y_1)
        else:
            ux=0
            uy=0

        x_1 = ex;
        y_1 = ey;

        cmd.twist.linear.x= uy*math.cos(orientation)+ux*math.sin(orientation)
        cmd.twist.linear.y= uy*math.sin(orientation)-ux*math.cos(orientation)

        cv2.imshow("Frame", frame)
        

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        pub.publish(cmd)
        rate.sleep()

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        tracker()
    except rospy.ROSInterruptException:
        pass