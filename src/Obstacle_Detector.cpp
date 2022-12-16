

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


#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
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

#include <pcl_obstacle_detection/Cloud_Filter.hpp>
#include <pcl_obstacle_detection/Obstacle_Detector.hpp>


// Dynamic parameter server callback function
void dynamicParamCallback(pcl_obstacle_detection::pcl_obstacle_detection_Config& config, uint32_t level)
{
  // Pointcloud Filtering Parameters
  rMin = config.Ground_Removal_rMin;
  rMax = config.Ground_Removal_rMax;
  tHmin = config.Ground_Removal_tHmin;
  tHmax = config.Ground_Removal_tHmax;
  tHDiff = config.Ground_Removal_tHDiff;
  hSensor = config.Ground_Removal_hSensor;
}

Obstacle_Detector::Obstacle_Detector()
{
    nh.getParam("/POINTCLOUD_TOPIC", POINTCLOUD_TOPIC);
    nh.getParam("/PUBLISH_GROUND", PUBLISH_GROUND);
    nh.getParam("/PUBLISH_CLUSTER", PUBLISH_CLUSTER);
    nh.getParam("/PUBLISH_BOX", PUBLISH_BOX);
    sub_lidar_points = nh.subscribe(POINTCLOUD_TOPIC, 1, &Obstacle_Detector::process_pc,this);
    pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_GROUND, 10);
    pub_cluster_cloud = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_CLUSTER, 10);
    pub_bounding_box = nh.advertise<visualization_msgs::MarkerArray>(PUBLISH_BOX,10);

    nh.getParam("/VoxelGridLeafSize",VoxelGridLeafSize);
    nh.getParam("/EuclideanClusterExtraction_DistanceThreshold",EuclideanClusterExtraction_DistanceThreshold);
    nh.getParam("/EuclideanClusterExtraction_ClusterTolerance",EuclideanClusterExtraction_ClusterTolerance);
    nh.getParam("/EuclideanClusterExtraction_MinClusterSize",EuclideanClusterExtraction_MinClusterSize);
    nh.getParam("/EuclideanClusterExtraction_MaxClusterSize",EuclideanClusterExtraction_MaxClusterSize);
    // raw_cloud = new pcl::PointCloud<pcl::PointXYZ>;

    // Dynamic Parameter Server & Function
    f = boost::bind(&dynamicParamCallback, _1, _2);
    server.setCallback(f);
}



void Obstacle_Detector::process_pc(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
    //
    ROS_DEBUG("Lidar Points Received");
    ros::Time lasttime = ros::Time::now();
    const auto pointcloud_header = cloud_msg->header;

    // Initialize PointCloud
    pcl::PointCloud<pcl::PointXYZ> raw_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    //Convert raw cloud to PCL format
    pcl::fromROSMsg(*cloud_msg,raw_cloud);
    // ROS_INFO("There are %ld points in raw_cloud",raw_cloud.size());
    
    // Removal ground
    groundRemove(raw_cloud, cluster_cloud, ground_cloud);
    
    //Find Clustering
    int numCluster = 0; //Number of Clustering
    std::array<std::array<int,numGrid>,numGrid> cartesianData{};
    componentClustering(cluster_cloud,cartesianData,numCluster);

    // ROS_INFO("Find %i clusters",numCluster);

    // To visualize the clustered cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr visulized_cluster_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    makeClusteredCloud(cluster_cloud,cartesianData,visulized_cluster_cloud);


    // Draw Bounding Boxes
    std::vector<pcl::PointCloud<pcl::PointXYZ>> bounding_boxes = boxFitting(cluster_cloud,cartesianData,numCluster);
    visualization_msgs::MarkerArray visual_bouding_box;
    // Draw_Bounding_Box()
    
    int ID = 0;
    if (bounding_boxes.size() != 0)
    {
        for (auto box : bounding_boxes)
        {
            visual_bouding_box.markers.push_back(Draw_Bounding_Box(box,pointcloud_header,ID));
            // visual_bouding_box.header = pointcloud_header;
            ID++;
        }  
    }
    ROS_INFO("find %i objects",ID);

    //Publish Ground cloud and obstacle cloud
    sensor_msgs::PointCloud2 cluster_cloud_msg;
    sensor_msgs::PointCloud2 ground_cloud_msg;
    pcl::toROSMsg(*visulized_cluster_cloud,cluster_cloud_msg);
    pcl::toROSMsg(*ground_cloud,ground_cloud_msg);
    cluster_cloud_msg.header = pointcloud_header;
    ground_cloud_msg.header = pointcloud_header;

    pub_ground_cloud.publish(ground_cloud_msg);
    pub_cluster_cloud.publish(cluster_cloud_msg);
    pub_bounding_box.publish(visual_bouding_box);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Obstacle_Detector::FilterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud) 
{
    const Eigen::Vector4f min_pt = Eigen::Vector4f(-30,-30,-2.5,1);
    const Eigen::Vector4f max_pt = Eigen::Vector4f(70,30,1,1);
    // Create the filtering object: downsample the dataset using a leaf size
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(input_cloud);
    vg.setLeafSize(VoxelGridLeafSize,VoxelGridLeafSize,VoxelGridLeafSize);
    vg.filter(*filtered_cloud);


    // Maybe do a cropbox


    //Ground Plane Segmentation
    // pcl::PointIndicesPtr ground (new pcl::PointIndices);
    // pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
    // pmf.setInputCloud(filtered_cloud);
    // pmf.setMaxWindowSize(20);
    // pmf.setSlope(1.0f);
    // pmf.setInitialDistance(0.5f);
    // pmf.setMaxDistance(3.0f);
    // pmf.extract(ground ->indices);

    // pcl::ExtractIndices<pcl::PointXYZ> extract;
    // extract.setInputCloud(filtered_cloud);
    // extract.setIndices(ground);
    // extract.setNegative(true);
    // extract.filter(*filtered_cloud);



    return filtered_cloud;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr Obstacle_Detector::SegmentCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
{
    // The smallest scale to use in the DoN filter
    double scale1 = 1;

    //The largest scale to use in the DoN filter
    double scale2 = 3;

    //The minimum DoN magnitude to threshold by
    double threshold = 0.15;

    //Segment scene into cluster with given distance tolerence using euclidean clustering
    double segradius = 2;

    //Create a search tree, use KDTree for non-organized data
    pcl::search::Search<pcl::PointXYZ>::Ptr tree;
    if (input_cloud -> isOrganized())
    {
        tree.reset (new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
        // std::cout<< "the input cloud is Organized" <<std::endl;
    }
    else
    {
        tree.reset (new pcl::search::KdTree<pcl::PointXYZ>(false));
        // std::cout<< "the input cloud is not Organized" <<std::endl;
    }

    //Set the input pointcloud for the search tree
    tree-> setInputCloud(input_cloud);

    // Compute normals using both small and large scale at each point
    pcl::NormalEstimationOMP<pcl::PointXYZ,pcl::PointNormal> ne;
    ne.setInputCloud(input_cloud);
    ne.setSearchMethod(tree);

    //Set view point to ensure the normals are all pointed in the same direction
    ne.setViewPoint(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max());

    //Calculate normals with the small scale
    pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale (new pcl::PointCloud<pcl::PointNormal>);
    ne.setRadiusSearch(scale1);
    ne.compute(*normals_small_scale);


    //Calculate normals with the large scale
    pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale (new pcl::PointCloud<pcl::PointNormal>);
    ne.setRadiusSearch(scale2);
    ne.compute(*normals_large_scale);

    //Create output cloud for DoN results
    pcl::PointCloud<pcl::PointNormal>::Ptr doncloud (new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*input_cloud,*doncloud);

    //Create DoN operator
    pcl::DifferenceOfNormalsEstimation<pcl::PointXYZ,pcl::PointNormal,pcl::PointNormal> don;
    don.setInputCloud(input_cloud);
    don.setNormalScaleLarge(normals_large_scale);
    don.setNormalScaleSmall(normals_small_scale);

    if (!don.initCompute())
    {
        std::cerr << "Error: Could not initialize DoN feature operator" << std::endl;
    }

    don.computeFeature(*doncloud);

    //Built the condition for filtering
    pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond (new pcl::ConditionOr<pcl::PointNormal>);
    range_cond-> addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(new pcl::FieldComparison<pcl::PointNormal>("curvature",pcl::ComparisonOps::GT,threshold)));

    //Build the filter
    pcl::ConditionalRemoval<pcl::PointNormal> condrem;
    condrem.setCondition(range_cond);
    condrem.setInputCloud(doncloud);

    pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered (new pcl::PointCloud<pcl::PointNormal>);

    //Apply filter
    condrem.filter(*doncloud_filtered);

    //TODO: cast to doncloud???
    doncloud = doncloud_filtered;


    // Fiilter by Magnitude
    pcl::search::KdTree<pcl::PointNormal>::Ptr segtree (new pcl::search::KdTree<pcl::PointNormal>);
    segtree->setInputCloud(doncloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointNormal> ec;

    ec.setClusterTolerance(segradius);
    ec.setMinClusterSize(5);
    ec.setMaxClusterSize(100000);
    ec.setSearchMethod(segtree);
    ec.setInputCloud(doncloud);
    ec.extract(cluster_indices);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_cluster_don (new pcl::PointCloud<pcl::PointNormal>);
    for (const auto& cluster : cluster_indices)
    {
        for (const auto& idx :cluster.indices)
        {
            cloud_cluster_don->points.push_back((*doncloud)[idx]);
        }
    }

    cloud_cluster_don -> width  = cloud_cluster_don -> size();
    cloud_cluster_don -> height  = 1;
    cloud_cluster_don -> is_dense = true;

    pcl::PointCloud<pcl::PointXYZ>::Ptr return_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud_cluster_don,*return_cloud);

    return return_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Obstacle_Detector::EuclideanClusterExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<pcl::PointXYZ> seg;
     pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (EuclideanClusterExtraction_DistanceThreshold);



    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (EuclideanClusterExtraction_ClusterTolerance); // 2cm
    ec.setMinClusterSize (EuclideanClusterExtraction_MinClusterSize);
    ec.setMaxClusterSize (EuclideanClusterExtraction_MaxClusterSize);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);

    ROS_INFO("cluster_indice size: %li",cluster_indices.size());
    int j=0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& cluster : cluster_indices)
    {
        for (const auto& idx : cluster.indices) 
        {
            cloud_out -> push_back((*cloud_filtered)[idx]);
        }
    }
    cloud_out->width = cloud_out -> size();
    cloud_out->height = 1;
    cloud_out->is_dense = true;

    return cloud_filtered;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr Obstacle_Detector::SegmentPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,const pcl::PointIndices::ConstPtr& inliers)
{

    //Pushback all the inliers into plane_cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ> ());

    for (int index : inliers->indices)
    {
        plane_cloud->points.push_back(input_cloud->points[index]);
    }

    return plane_cloud;

}


pcl::PointCloud<pcl::PointXYZ>::Ptr Obstacle_Detector::SegmentCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,const pcl::PointIndices::ConstPtr& inliers)
{
    //extract all points that are not in the inliers to cluster_cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cluster_cloud);

    return cluster_cloud;

}