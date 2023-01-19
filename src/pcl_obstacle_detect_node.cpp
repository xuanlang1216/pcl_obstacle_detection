// Include the ROS library
#include <ros/ros.h>
#include <ros/console.h>

// // for pcd conversion
// #include <chrono>
// #include <thread>

// Include pcl
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/don.h>
#include <pcl/conversions.h>

// Include PointCloud2 message
#include <sensor_msgs/PointCloud2.h>

//Include custom library
#include <pcl_obstacle_detection/Obstacle_Detector.hpp>
#include <pcl_obstacle_detection/Object_Tracking.hpp>


// Main function
int main(int argc, char** argv)
{
    // Initialize the ROS Node "pcl_obstacle_detection"
    ros::init(argc, argv, "pcl_obstacle_detection");

    // // Instantiate the ROS Node Handler as nh
    // ros::NodeHandle nh;

    // Print "Hello ROS!" to the terminal and ROS log file
    ROS_INFO_STREAM("Hello from ROS node " << ros::this_node::getName());

    // Obstacle_Detector my_detector;
    //testing
    ObjectTracker my_tracker(3.0,4.0);
    my_tracker.statePrediction(0.2);
    Eigen::MatrixXd m(2,1);
    m(0,0) = 3.1;
    m(1,0) = 4.1;
    my_tracker.stateUpdate(m,0.2);

    // Create a ROS Subscriber to IMAGE_TOPIC with a queue_size of 1 and a callback function to cloud_cb
    // ros::Subscriber sub = nh.subscribe(POINTCLOUD_TOPIC, 1, process_pc);


    // // Create a ROS publisher to PUBLISH_TOPIC with a queue_size of 1
    // pub = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_TOPIC, 1);

    // pub_cluster = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_CLUSTER,1);



    // Spin
    ros::spin();

    // Program succesful
    return 0;
}