#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Visualiser:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ln, = plt.plot([], [], 'ro')
        self.ln2, = plt.plot([],[],'go')
        self.list_update = {} 
        self.tracking_x ={}
        self.tracking_y ={}
        self.colors = ['bo','go','']
        self.x_data, self.y_data = np.array([]) , np.array([]) 

    def plot_init(self):
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        return self.ln

    def callback(self,Markers:MarkerArray):
        rospy.loginfo("find %i object",Markers.markers.__len__())
        

        for marker in Markers.markers:
            # print()
            id = marker.id
            if id in self.tracking_x.keys():
                self.tracking_x[id] = np.append(self.tracking_x[id],marker.pose.position.x)
                self.tracking_y[id] = np.append(self.tracking_y[id],marker.pose.position.y)
            else:
                self.tracking_x[id] = np.array([marker.pose.position.x])
                self.tracking_y[id] = np.array([marker.pose.position.y])


            # rospy.loginfo("ID: %i",marker.id)

    
    
    def update_plot(self, frame):
        
        for key in self.tracking_x:
            if key in self.list_update.keys():
                self.list_update[key].set_data(self.tracking_x[key],self.tracking_y[key])
            else:
                self.list_update[key], = plt.plot(self.tracking_x[key], self.tracking_y[key], color ='blue')

        print()
        return tuple(list(self.list_update.values()))
            




rospy.init_node('listener',anonymous= True)
vis = Visualiser()
rospy.Subscriber("/pcl/tracker",MarkerArray,vis.callback)
ani = FuncAnimation(vis.fig,vis.update_plot,init_func=vis.plot_init)
plt.show(block = True)
# rospy.spin()




        
