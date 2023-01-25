# pragma once

// Include the ROS library
#include <ros/ros.h>
#include <ros/console.h>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>

// Include PointCloud2 message
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

//Dynamic Reconfig
#include <dynamic_reconfigure/server.h>
#include <pcl_obstacle_detection/pcl_obstacle_detection_Config.h>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/progressive_morphological_filter.h>

#include <pcl/features/don.h>

#include <pcl_obstacle_detection/Object_Tracking.hpp>


class Obstacle_Detector
{
    private:
        /* Parameter */
        float VoxelGridLeafSize;
        double EuclideanClusterExtraction_DistanceThreshold;
        double EuclideanClusterExtraction_ClusterTolerance;
        int EuclideanClusterExtraction_MinClusterSize;
        int EuclideanClusterExtraction_MaxClusterSize;

        double lasttime;



        // ROS Publisher
        ros::NodeHandle nh;
        ros::Subscriber sub_lidar_points;
        ros::Publisher pub_ground_cloud;
        ros::Publisher pub_cluster_cloud;
        ros::Publisher pub_bounding_box;
        ros::Publisher pub_trakers;

        std::string POINTCLOUD_TOPIC;
        std::string PUBLISH_GROUND;
        std::string PUBLISH_CLUSTER;
        std::string PUBLISH_BOX;
        std::string PUBLISH_TRACKER;

        std::vector<ObjectTracker> Objects;

        // TODO: Find a way to use this
        // pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud;
        // pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud;
        // pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud;
        // pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud;

        dynamic_reconfigure::Server<pcl_obstacle_detection::pcl_obstacle_detection_Config> server;
        dynamic_reconfigure::Server<pcl_obstacle_detection::pcl_obstacle_detection_Config>::CallbackType f;
        
    public:
        Obstacle_Detector();
        // ~Obstacle_Detector();

        // void dynamicParamCallback(pcl_obstacle_detection::pcl_obstacle_detection_Config& config, uint32_t level);

        void process_pc(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg);
        void updateTracker(pcl::PointCloud<pcl::PointXYZ>& object,double time_diff);
        visualization_msgs::MarkerArray Draw_Trackers(std_msgs::Header header);




        pcl::PointCloud<pcl::PointXYZ>::Ptr FilterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr SegmentCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr EuclideanClusterExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr SegmentPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,const pcl::PointIndices::ConstPtr& inliers);
        pcl::PointCloud<pcl::PointXYZ>::Ptr SegmentCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,const pcl::PointIndices::ConstPtr& inliers);
        
};

