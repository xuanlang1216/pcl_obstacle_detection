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
        self.fig_trag = plt.figure('Trajectories')
        self.ax_trag = self.fig_trag.add_subplot()

        self.fig_data = plt.figure('Data Visualization')
        self.ax_x = self.fig_data.add_subplot(211)
        self.ax_y = self.fig_data.add_subplot(212)
        # self.ax_xvel = self.fig_data.add_subplot()
        # self.ax_yvel = self.fig_data.add_subplot()

        self.xy_plot = {} # [key,plot]
        self.x_plot = {}
        self.y_plot = {}
        self.xvel_plot = {}
        self.yvel_plot = {}
        self.time = {} # time for each individual object
        self.tracking_x ={}
        self.tracking_y ={}
        self.tracking_xvel = {}
        self.tracking_yvel = {}
        self.colors = {}
        self.legend_trag = self.ax_trag.legend()
        self.legend_x = self.ax_x.legend()
        # self.legend_y = self.ax_y.legend()

        self.start_time = 0.0

    def plot_init_trag(self):
        self.ax_trag.set_xlim(-20, 20)
        self.ax_trag.set_ylim(-20, 20)
        self.ax_trag.grid()
        return self.ax_trag

    def plot_init_data(self):
        self.ax_x.set_xlim(0,80)
        self.ax_x.set_ylim(-20,20)
        self.ax_x.grid()

        self.ax_y.set_xlim(0,80)
        self.ax_y.set_ylim(-20,20)
        self.ax_y.grid()

        # self.ax_xvel.set_xlim(-20,20)
        # self.ax_xvel.set_ylim(-20,20)
        # self.ax_xvel.grid()

        # self.ax_yvel.set_xlim(-20,20)
        # self.ax_yvel.set_ylim(-20,20)
        # self.ax_yvel.grid()
        return self.ax_x #, self.ax_y,self.ax_xvel,self.yvel

    def callback(self,Markers:MarkerArray):
        rospy.loginfo("find %i object",Markers.markers.__len__())

        # if np.any(self.time):
        #     self.time = np.append(self.time,rospy.get_time()-self.start_time)
        # else:
        #     self.start_time = rospy.get_time()
        #     self.time = np.append(self.time,rospy.get_time()-self.start_time)

        if self.time == {}:
            self.start_time = rospy.get_time()

        

        for marker in Markers.markers:
            # print()
            id = marker.id
            if id in self.tracking_x.keys():
                self.tracking_x[id] = np.append(self.tracking_x[id],marker.pose.position.x)
                self.tracking_y[id] = np.append(self.tracking_y[id],marker.pose.position.y)
                self.time[id] = np.append(self.time[id],rospy.get_time()-self.start_time)

            else:
                self.tracking_x[id] = np.array([marker.pose.position.x])
                self.tracking_y[id] = np.array([marker.pose.position.y])
                self.time[id] = np.array([rospy.get_time()-self.start_time])
                self.colors[id] = '#%06x'%np.random.randint(0,0xFFFFFF)


            # rospy.loginfo("ID: %i",marker.id)

    
    
    def update_plot_trag(self, frame):
        
        for key in self.tracking_x:
            if key in self.xy_plot.keys():
                self.xy_plot[key].set_data(self.tracking_x[key],self.tracking_y[key])
            else:
                self.xy_plot[key], = self.ax_trag.plot(self.tracking_x[key], self.tracking_y[key], color = self.colors[key])
                self.legend_trag.remove()
                self.legend_trag = self.ax_trag.legend(list(self.xy_plot.keys()))

        return tuple(list(self.xy_plot.values())) ,self.legend_trag
    
    def update_plot_data(self,frame):
        # print(self.xy_plot)
        for key in self.tracking_x:
            if key in self.x_plot.keys():
                self.x_plot[key].set_data(self.time[key],self.tracking_x[key])
                
                self.y_plot[key].set_data(self.time[key],self.tracking_y[key])

            else:
                self.x_plot[key], = self.ax_x.plot(self.time[key], self.tracking_x[key], color = self.colors[key])
                
                self.y_plot[key], = self.ax_y.plot(self.time[key], self.tracking_y[key], color = self.colors[key])

                self.legend_x.remove()
                self.legend_x = self.ax_x.legend(list(self.x_plot.keys()))

        
        return tuple(list(self.x_plot.values())) ,self.legend_x




rospy.init_node('listener',anonymous= True)
vis = Visualiser()
rospy.Subscriber("/pcl/tracker",MarkerArray,vis.callback)
ani = FuncAnimation(vis.fig_trag,vis.update_plot_trag,init_func=vis.plot_init_trag)
ani2 = FuncAnimation(vis.fig_data,vis.update_plot_data,init_func=vis.plot_init_data)
plt.show(block =True)
# rospy.spin()




        
