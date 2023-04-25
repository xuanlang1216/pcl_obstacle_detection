# LiDAR Obstacle Detection and Tracking


## Requirement
1.[PCL](https://github.com/PointCloudLibrary/pcl)

2.[jsk-recognition-msgs](http://wiki.ros.org/jsk_recognition_msgs)

3.[dynamic_reconfigure](http://wiki.ros.org/dynamic_reconfigure)




## Installation

```
// create catkin workspace and src folder
mkdir -p catkin_ws/src
cd catkin_ws/src
// git clone the repo
git clone https://github.com/xuanlang1216/pcl_obstacle_detection.git
cd ..

# build the package
catkin_make
source devel/setup.bash
```

## Example Run

Different .launch file contains different config for different sensor.

```
roslaunch pcl_obstacle_detection kitti.launch
```



## Run with Velodyne Sensor in Real Time
Follow [this ROS tutorial](http://wiki.ros.org/velodyne/Tutorials/Getting%20Started%20with%20the%20Velodyne%20VLP16). 

In short, you need [velodyne ros-driver](https://github.com/ros-drivers/velodyne). And run :

```
roslaunch velodyne_pointcloud VLP16_points.launch
```

to convert raw velodyne sensor message to pointcloud2 message