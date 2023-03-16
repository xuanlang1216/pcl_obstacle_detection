

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
    nh.getParam("/PUBLISH_TRACKER",PUBLISH_TRACKER);
    nh.getParam("/PUBLISH_OBJECT",PUBLISH_OBJECT);

    sub_lidar_points = nh.subscribe(POINTCLOUD_TOPIC, 1, &Obstacle_Detector::processPointCloud,this);
    pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_GROUND, 10);
    pub_cluster_cloud = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_CLUSTER, 10);
    pub_bounding_box = nh.advertise<visualization_msgs::MarkerArray>(PUBLISH_BOX,10);
    pub_trakers = nh.advertise<visualization_msgs::MarkerArray>(PUBLISH_TRACKER,10);
    pub_objects = nh.advertise<pcl_obstacle_detection::StampedObjectArray>(PUBLISH_OBJECT,10);

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

    ID_count = 1;

    // Dynamic Parameter Server & Function
    f = boost::bind(&dynamicParamCallback, _1, _2);
    server.setCallback(f);
}



void Obstacle_Detector::processPointCloud(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
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
    ROS_INFO("Find %i Clustered Objects",ID);

    // Tracking Update
    updateTrackers(centroids,time_now);

    visualization_msgs::MarkerArray visual_traker = drawTrackers(pointcloud_header);

    //pack the objects:
    pcl_obstacle_detection::StampedObjectArray publish_objects = packObjects(pointcloud_header);

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
    pub_objects.publish(publish_objects);

}

void Obstacle_Detector::updateTrackers(std::vector<Eigen::Vector4f> centroids,double time_now){


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
                Objects[min_dist_ID].UKFUpdate(measure,time_now);
                // std::cout<<"new_Position: (" << Objects[min_dist_ID].states_now_(0,0)<<", "<<Objects[min_dist_ID].states_now_(1,0)<<")"<<std::endl;
            }
            else if (min_dist >= 1 /*&& Objects.size() < 10*/){ //added object size constraints TODO: need to remove this later
                std::cout<<"Adding new traker Because MinDist = "<<min_dist <<std::endl;
                Objects.push_back(ObjectTracker(x,y,time_now,ID_count));
                ID_count += 1;
            }

        }
        else{
            std::cout<<"Adding new traker"<<std::endl;
            Objects.push_back(ObjectTracker(x,y,time_now,ID_count));
            ID_count += 1;
        }
    }
    
    // for other objects, do a state propagation
    for(int i =0; i<Objects.size();i++){
        if (!Objects[i].updated && Objects[i].tracking_state == 2){  // if the object did not get updated this frame but got updated last frame
            double time_diff = time_now - Objects[i].prev_time_;
            Objects[i].statePropagateOnly(time_diff);
        }
        
        Objects[i].updated = false; //reset the updated state
    }
    
    // Remove all objects that have not been updated for 5 sec
    auto erase_loss_object = [time_now](ObjectTracker i){return time_now - i.prev_time_ > 5.0;};
    auto find_loss_object = std::remove_if(Objects.begin(),Objects.end(),erase_loss_object);
    Objects.erase(find_loss_object,Objects.end());

    ROS_INFO("Tracking %i Objects",Objects.size());
    // std::cout<<"object count: " << Objects.size()<<std::endl;
}

visualization_msgs::Marker creatVelocityMarker (int ID,std_msgs::Header header,float x_pos,float y_pos,float x_vel,float y_vel){
    visualization_msgs::Marker marker;
    
    marker.ns = "velocity";
    marker.id = ID;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;

    marker.header = header;

    // marker.pose.position.x = x_pos;
    // marker.pose.position.y = y_pos;
    // marker.pose.position.z = -1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;


    double arrow_length = 5.0;
    double distance = sqrt(pow(x_vel,2)+pow(y_vel,2));
    double x_len = 5 * x_vel / distance;
    double y_len = 5 * y_vel / distance;
    


    geometry_msgs::Point start;
    start.x = x_pos;
    start.y = y_pos;
    start.z = -1;

    geometry_msgs::Point end;
    end.x = x_pos + x_len;
    end.y = y_pos + y_len;
    end.z = -1;
    
    marker.points = std::vector<geometry_msgs::Point>{start,end};
    
    marker.scale.x = 0.1;
    marker.scale.y = 0.5;
    marker.scale.z = 0.5;
    marker.color.r = 255.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0;


    marker.color.a = 1.0;

    marker.lifetime = ros::Duration(1);

    return marker;
}

visualization_msgs::Marker createPathMarker(int ID,std_msgs::Header header,std::vector<double> x_path,std::vector<double> y_path){
    
    visualization_msgs::Marker marker;
    marker.ns = "Path";
    marker.id = ID;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.points = std::vector<geometry_msgs::Point>{};
    marker.header = header;

    for (int i =0; i<x_path.size();i++){
        geometry_msgs::Point temp;
        temp.x = x_path[i];
        temp.y = y_path[i];
        temp.z = -1;
        marker.points.push_back(temp);
    }

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.color.r = 0.0f;
    marker.color.g = 0.0f;
    marker.color.b = 255.0f;
    marker.color.a = 1.0;
    marker.lifetime = ros::Duration(1);

    return marker;
}

visualization_msgs::MarkerArray Obstacle_Detector::drawTrackers(std_msgs::Header header)
{
    visualization_msgs::MarkerArray visual_objects;
    // add Bounding box
    // ROS_INFO("%i objects to draw",Objects.size());
    for (auto object : Objects)
    {
        float x_pos = object.states_now_(0,0);
        float y_pos = object.states_now_(1,0);
        float x_vel = object.states_now_(2,0);
        float y_vel = object.states_now_(3,0);
        
        // only draw velocity that is greater than 0.1
        if (sqrt(pow(x_vel,2)+pow(y_vel,2)) > 0.1){
            // ROS_INFO("object %li have small vel: x_vel: %f, y_vel: %f",object.ID,x_vel,y_vel);
            visual_objects.markers.push_back(creatVelocityMarker(object.ID,header,x_pos,y_pos,x_vel,y_vel));
        }

        //add marker for pathes
        if (object.x_history.size() == object.y_history.size() && object.x_history.size()>1){
            visual_objects.markers.push_back(createPathMarker(object.ID,header,object.x_history,object.y_history));
        }
        else{
            ROS_ERROR("Path have different length!");
        }
        

        uint32_t shape = visualization_msgs::Marker::CUBE;
        visualization_msgs::Marker marker;

        marker.ns = "cube";
        marker.id = object.ID;
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

        marker.lifetime = ros::Duration(1);

        visual_objects.markers.push_back(marker);


        // add text for speed and position
        visualization_msgs::Marker marker_text;
        marker_text.ns = "basic_shapes";
        marker_text.action = visualization_msgs::Marker::ADD;
        marker_text.id =object.ID;
        marker_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker_text.header = header;

        marker_text.scale.z = 0.2;
        marker_text.color.b = 0;
        marker_text.color.g = 0;
        marker_text.color.r = 255;
        marker_text.color.a = 1;

        marker_text.text = "    ID: " +std::to_string(object.ID)+"\nx: " + std::to_string(x_pos).substr(0,5) + " y: " + std::to_string(y_pos).substr(0,5) +"\nx_vel: " + std::to_string(x_vel).substr(0,5) + " y_vel:" + std::to_string(y_vel).substr(0,5);

        marker_text.pose.position.x = object.states_now_(0,0);
        marker_text.pose.position.y = object.states_now_(1,0);
        marker_text.pose.position.z = 1.0;
        marker_text.pose.orientation.x = 0.0;
        marker_text.pose.orientation.y = 0.0;
        marker_text.pose.orientation.z = 0.0;
        marker_text.pose.orientation.w = 1.0;
        marker_text.scale.x = 0.3;
        marker_text.scale.y = 0.3;
        marker_text.scale.z = 0.3;
        marker_text.lifetime = ros::Duration(1);
        visual_objects.markers.push_back(marker_text);
    }

    return visual_objects;
    
}


pcl_obstacle_detection::StampedObjectArray Obstacle_Detector::packObjects(std_msgs::Header header){
    pcl_obstacle_detection::StampedObjectArray publish_objects;

    for (auto object : Objects){
        pcl_obstacle_detection::StampedObject the_object;

        the_object.pose.position.x = object.states_now_(0,0);
        the_object.pose.position.y = object.states_now_(1,0);

        the_object.velocity.linear.x = object.states_now_(2,0);
        the_object.velocity.linear.y = object.states_now_(3,0);

        the_object.ID = object.ID;

        the_object.header = header;

        // TODO: change to actual value
        the_object.height = 10;
        the_object.width = 10;
        the_object.length = 10;

        the_object.covariances = std::vector<float>(object.P_now_.data(), object.P_now_.data() +object.P_now_.size());
        
        // std::cout<<"casted:"<<std::endl;
        // std::for_each(the_object.covariances.begin(),
        //         the_object.covariances.end(),
        //         [](const auto &e) {
        //             std::cout << e << " ";
        //         });

        // std::cout<<"\nactual:"<<std::endl;
        // std::cout<<object.P_now_<<std::endl;
        

        publish_objects.objects.push_back(the_object);


    }

    return publish_objects;
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