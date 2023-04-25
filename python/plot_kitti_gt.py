#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
# from StampedObjectArray.msg import StampedObjectArray

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def processTracklet(input:MarkerArray):
    global tracklet_ID,tracklet_time,tracklet_x,tracklet_y,time_initial
    for marker in input.markers:
        if marker.ns == 'box' and marker.id == tracklet_ID:
            time_now = rospy.Time.now().to_sec()-time_initial
            tracklet_time.append(time_now)
            tracklet_x.append(marker.pose.position.x)
            tracklet_y.append(marker.pose.position.y)
            print('find tracklet:',time_now,marker.pose.position.x,marker.pose.position.y)


def processTracker(input:MarkerArray):
    global tracker_ID,tracker_time,tracker_x,tracker_y,time_initial
    for marker in input.markers:
        if marker.ns == 'cube' and marker.id == tracker_ID:
            time_now = rospy.Time.now().to_sec()-time_initial
            tracker_time.append(time_now)
            tracker_x.append(marker.pose.position.x)
            tracker_y.append(marker.pose.position.y)
            print('find tracker:',time_now, marker.pose.position.x,marker.pose.position.y)

if __name__ == "__main__":
    

    tracklet_ID = 8
    tracker_ID = 238
    
    tracklet_time = []
    tracklet_x = []
    tracklet_y = []

    tracker_time = []
    tracker_x = []
    tracker_y = []

    framecount = 0 
    # Tracklet_Dir = 'tracklet_labels.xml'
    # tracklet = parseXML(Tracklet_Dir)
    # print(len(tracklet))
    rospy.init_node('plotter',anonymous= True)
    time_initial = rospy.Time.now().to_sec()
    sub_object = rospy.Subscriber("kitti_tracklet",MarkerArray,processTracklet)

    rospy.Subscriber("/pcl/tracker",MarkerArray,processTracker)
    pub_kitti_tracklet = rospy.Publisher('kitti_tracklet',MarkerArray,queue_size=10)

    # rospy.spin()
    
    while True:
        if (rospy.Time.now().to_sec()-time_initial) > 40:
            fig1, ax1 = plt.subplots()

            # Plot the data
            ax1.plot(tracklet_x, tracklet_y,label = 'Ground Truth')

            ax1.plot(tracker_x, tracker_y,label = 'Tracking Position')

            # Set the axis labels
            ax1.set_xlabel('X Position [m]')
            ax1.set_ylabel('Y Position [m]')

            # Set the title
            ax1.set_title('Object Tracking XY Plot [Object 8]')

            ax1.legend(['Ground Truth','Tracking Position'])



            # time vs x,y plot
            fig2,(xplt,yplt) = plt.subplots(nrows=2, ncols=1)
            xplt.plot(tracklet_time,tracklet_x,label ='Ground Truth')
            xplt.plot(tracker_time,tracker_x,label ='Tracking Position')
            xplt.set_xlabel('Time [s]')
            xplt.set_ylabel('X Position [m]')
            xplt.set_title('X vs. Time [Object 8]')
            xplt.legend(['Ground Truth','Tracking Position'])


            yplt.plot(tracklet_time,tracklet_y,label ='Ground Truth')
            yplt.plot(tracker_time,tracker_y,label ='Tracking Position')
            yplt.set_xlabel('Time [s]')
            yplt.set_ylabel('Y Position [m]')
            yplt.set_title('Y vs. Time [Object 8]')
            yplt.legend(['Ground Truth','Tracking Position'])


            # Show the plot
            plt.show()

            break

    rospy.spin()    

