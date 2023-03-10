

#include <ros/ros.h>
#include <ros/console.h>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <algorithm>

// Include PointCloud2 message
#include <sensor_msgs/PointCloud2.h>



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

    tHeightMin = config.BoxFit_tHeightMin;  //Min height of the object
    tHeightMax = config.BoxFit_tHeightMax;  //max height of the object
    tWidthMin = config.BoxFit_tWidthMin;  //min Width of the object
    tWidthMax = config.BoxFit_tWidthMax;   //max wdith of the object
    tLenMin = config.BoxFit_tLenMin;     //min length of the object
    tLenMax = config.BoxFit_tLenMax;    //max length of the object
    tAreaMax = config.BoxFit_tAreaMax;   //max area of the object
    tRatioMin = config.BoxFit_tRatioMin;   //min ratio between length and width
    tRatioMax = config.BoxFit_tRatioMax;   //max ratio between length and width
    minLenRatio = config.BoxFit_minLenRatio; //min length of object for ratio check
    tPtPerM3 = config.BoxFit_tPtPerM3;      //min point count per bouding box volume

    roiM = config.ComponentClustering_roiM;

}

Obstacle_Detector::Obstacle_Detector()
{
    nh.getParam("/POINTCLOUD_TOPIC", POINTCLOUD_TOPIC);
    nh.getParam("/PUBLISH_GROUND", PUBLISH_GROUND);
    nh.getParam("/PUBLISH_CLUSTER", PUBLISH_CLUSTER);
    nh.getParam("/PUBLISH_BOX", PUBLISH_BOX);
    nh.getParam("PUBLISH_TRACKER",PUBLISH_TRACKER);

    sub_lidar_points = nh.subscribe(POINTCLOUD_TOPIC, 1, &Obstacle_Detector::process_pc,this);
    pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_GROUND, 10);
    pub_cluster_cloud = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_CLUSTER, 10);
    pub_bounding_box = nh.advertise<visualization_msgs::MarkerArray>(PUBLISH_BOX,10);
    pub_trakers = nh.advertise<visualization_msgs::MarkerArray>(PUBLISH_TRACKER,10);

    nh.getParam("/VoxelGridLeafSize",VoxelGridLeafSize);
    nh.getParam("/EuclideanClusterExtraction_DistanceThreshold",EuclideanClusterExtraction_DistanceThreshold);
    nh.getParam("/EuclideanClusterExtraction_ClusterTolerance",EuclideanClusterExtraction_ClusterTolerance);
    nh.getParam("/EuclideanClusterExtraction_MinClusterSize",EuclideanClusterExtraction_MinClusterSize);
    nh.getParam("/EuclideanClusterExtraction_MaxClusterSize",EuclideanClusterExtraction_MaxClusterSize);
    // raw_cloud = new pcl::PointCloud<pcl::PointXYZ>;

   nh.getParam("GroundRemoval_rMin",rMin);
   nh.getParam("GroundRemoval_rMax",rMax);
   nh.getParam("GroundRemoval_tHmin",tHmin);
   nh.getParam("GroundRemoval_tHmax",tHmax);
   nh.getParam("GroundRemoval_tHDiff",tHDiff);
   nh.getParam("GroundRemoval_hSensor",hSensor);

   nh.getParam("ComponentClustering_roiM",roiM);
   nh.getParam("ComponentClustering_kernelSize",kernelSize);


   nh.getParam("BoxFitting_tHeightMin",tHeightMin);
   nh.getParam("BoxFitting_tHeightMax",tHeightMax);
   nh.getParam("BoxFitting_tWidthMin",tWidthMin);
   nh.getParam("BoxFitting_tWidthMax",tWidthMax);
   nh.getParam("BoxFitting_tLenMin",tLenMin);
   nh.getParam("BoxFitting_tAreaMax",tAreaMax);
   nh.getParam("BoxFitting_tRatioMin",tRatioMin);
   nh.getParam("BoxFitting_tRatioMax",tRatioMax);
   nh.getParam("BoxFitting_minLenRatio",minLenRatio);
   nh.getParam("BoxFitting_tPtPerM3",tPtPerM3);

    // Dynamic Parameter Server & Function
    f = boost::bind(&dynamicParamCallback, _1, _2);
    server.setCallback(f);
}



void Obstacle_Detector::process_pc(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
    //
    ROS_DEBUG("Lidar Points Received");
    double time_diff = ros::Time::now().toSec() - lasttime;
    std::cout<<"time_diff: "<<time_diff<<std::endl;
    double time_now = ros::Time::now().toSec();
    lasttime = ros::Time::now().toSec();
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

    ROS_INFO("Find %i clusters",numCluster);

    // To visualize the clustered cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr visulized_cluster_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    makeClusteredCloud(cluster_cloud,cartesianData,visulized_cluster_cloud);


    // Draw Bounding Boxes
    std::vector<pcl::PointCloud<pcl::PointXYZ>> bounding_boxes = boxFitting(cluster_cloud,cartesianData,numCluster);
    visualization_msgs::MarkerArray visual_bouding_box;

    //store the measurmeent distance
    std::vector<Eigen::Vector4f> centroids;

    // Draw_Bounding_Box()
    int ID = 0;
    if (bounding_boxes.size() != 0)
    {
        for (auto box : bounding_boxes)
        {
            visual_bouding_box.markers.push_back(Draw_Bounding_Box(box,pointcloud_header,ID));

            //calculate centroid
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(box,centroid);
            // std::cout<<centroid<<std::endl;
            centroids.push_back(centroid);
            // std::cout<<centroids[ID]<<std::endl;
            //Update the tracker
            // updateTracker(box,time_diff);
            // visual_bouding_box.header = pointcloud_header;
            ID++;
        }  
    }
    ROS_INFO("find %i objects",ID);

    // Tracking Update
    updateTrackers(centroids,time_diff);

    visualization_msgs::MarkerArray visual_traker = Draw_Trackers(pointcloud_header);

    //Update the tracker

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
    pub_trakers.publish(visual_traker);

}

void Obstacle_Detector::updateTrackers(std::vector<Eigen::Vector4f> centroids,double time_diff){


    for (auto& centroid:centroids)
    {
        double x = centroid[0];
        double y = centroid[1];
        double min_dist = 1000000;
        int min_dist_ID = -1;
        

        if (Objects.size()!=0){
            for (int i =0; i<Objects.size();i++){
                double dist = sqrt(pow(Objects[i].states_now_(0,0)-x,2) + pow(Objects[i].states_now_(1,0)-y,2));
                if (dist < min_dist){
                    min_dist = dist;
                    min_dist_ID = i;
                }
            }
            
            // std::cout<< "Min_dist: "<< min_dist << ", ID: "<< min_dist_ID<<std::endl;
            if (min_dist < 2 && min_dist_ID >= 0){ 
                // std::cout<<"Object(" << min_dist_ID<<") is updated"<<std::endl;
                // std::cout<<"Measurement :( " <<x<< " ,"<<y<<")"<<std::endl;
                // std::cout<<"Prev_Position: (" << Objects[min_dist_ID].states_now_(0,0)<<", "<<Objects[min_dist_ID].states_now_(1,0)<<")"<<std::endl;
                Eigen::MatrixXd measure(2,1);
                measure(0,0) = x;
                measure(1,0) = y;
                Objects[min_dist_ID].UKFUpdate(measure,time_diff);
                // std::cout<<"new_Position: (" << Objects[min_dist_ID].states_now_(0,0)<<", "<<Objects[min_dist_ID].states_now_(1,0)<<")"<<std::endl;
            }
            else if (min_dist >= 1 /*&& Objects.size() < 10*/){ //added object size constraints TODO: need to remove this later
                std::cout<<"Adding new traker Because MinDist = "<<min_dist <<std::endl;
                Objects.push_back(ObjectTracker(x,y,time_diff));
            }

        }
        else{
            std::cout<<"Adding new traker"<<std::endl;
            Objects.push_back(ObjectTracker(x,y,time_diff));
        }
    }

    // for other objects, do a state propagation
    for(int i =0; i<Objects.size();i++){
        if (!Objects[i].updated && Objects[i].tracking_state == 2){  // if the object did not get updated this frame but got updated last frame
            Objects[i].statePropagateOnly(time_diff);
        }
        
        Objects[i].updated = false; //reset the updated state

    }


    std::cout<<"object count: " << Objects.size()<<std::endl;
}

visualization_msgs::Marker creatVelocityMarker (uint32_t shape,int ID,std::string ns,std_msgs::Header header,float x_pos,float y_pos,float vel){
    visualization_msgs::Marker marker;
    
        marker.ns = ns;
        marker.id = ID;
        marker.type = shape;
        marker.action = visualization_msgs::Marker::ADD;

        marker.header = header;

        marker.pose.position.x = x_pos;
        marker.pose.position.y = y_pos;
        marker.pose.position.z = -1.0;
        

        if (ns.compare("x_vel")){
            marker.pose.orientation.x = 1.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 0.0;
            marker.scale.x = vel;
            marker.scale.y = 0.5;
            marker.scale.z = 0.5;
            marker.color.r = 255.0f;
            marker.color.g = 0.0f;
            marker.color.b = 0.0;
        }
        else if (ns.compare("y_vel")){
            marker.pose.orientation.x = 0.707;
            marker.pose.orientation.y = 0.707;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 0.0;
            marker.scale.x = vel;
            marker.scale.y = 0.5;
            marker.scale.z = 0.5;
            marker.color.r = 0.0f;
            marker.color.g = 0.0f;
            marker.color.b = 255.0f;
        }
        else{
            ROS_ERROR("Velocity Marker Wrong NameSpace (ns): %s",ns.data());
        }


        marker.color.a = 1.0;

        marker.lifetime = ros::Duration(500);

    return marker;


}

visualization_msgs::MarkerArray Obstacle_Detector::Draw_Trackers(std_msgs::Header header)
{
    visualization_msgs::MarkerArray visual_objects;
    int ID = 1;
    for (auto object : Objects)
    {
        float x_pos = object.states_now_(0,0);
        float y_pos = object.states_now_(1,0);
        float x_vel = object.states_now_(2,0);
        float y_vel = object.states_now_(3,0);
        
        // if (x_vel != 0){
        //     visual_objects.markers.push_back(creatVelocityMarker(0,ID,"x_vel",header,x_pos,y_pos,x_vel));
        // }
        // if (y_vel != 0){
        //     visual_objects.markers.push_back(creatVelocityMarker(0,ID,"y_vel",header,x_pos,y_pos,y_vel));
        // }
        

        uint32_t shape = visualization_msgs::Marker::CUBE;
        visualization_msgs::Marker marker;

        marker.ns = "cube";
        marker.id = ID;
        marker.type = shape;
        marker.action = visualization_msgs::Marker::ADD;

        marker.header = header;

        marker.pose.position.x = object.states_now_(0,0);
        marker.pose.position.y = object.states_now_(1,0);
        marker.pose.position.z = -1.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = 1;
        marker.scale.y = 1;
        marker.scale.z = 1;
        
        if (object.tracking_state == 1){ //just initialized
            marker.color.r = 255.0f;
            marker.color.g = 0.0f;
            marker.color.b = 0.0;
        }
        else if(object.tracking_state == 2){ //got updated this frame
            marker.color.r = 0.0f;
            marker.color.g = 255.0f;
            marker.color.b = 0.0f;
        }
        else if(object.tracking_state == 3){ //no update this frame
            marker.color.r = 0.0f;
            marker.color.g = 0.0f;
            marker.color.b = 255.0f;
        }

        marker.color.a = 1.0;

        marker.lifetime = ros::Duration(500);

        visual_objects.markers.push_back(marker);

        ID++;

    }

    

    return visual_objects;
    
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