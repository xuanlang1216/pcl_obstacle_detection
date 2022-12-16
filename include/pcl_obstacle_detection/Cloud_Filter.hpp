#ifndef MY_CLOUD_FILTER_H
#define MY_CLOUD_FILTER_H

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <vector>
#include <array>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

/*
Variables for Ground Removal
*/
const int numChannel = 80;
const int numBin = 120;
//const int numMedianKernel = 1;
extern float rMin;
extern float rMax;
//const float tHmin = -2.15;
extern float tHmin;
extern float tHmax;
//const float tHDiff = 0.3;
// since estimated ground plane = -1.73 by sensor height,
// tMin = -2.0
extern float tHDiff;
extern float hSensor;

/*
Variables for Component Clustering
*/
const int numGrid = 300;
extern float roiM;
extern int kernelSize;

/*
Variables for Box Fitting
*/
extern float picScale;
extern int ramPoints;
extern int lSlopeDist;
extern int lnumPoints;

extern float tHeightMin;
extern float tHeightMax;
extern float tWidthMin;
extern float tWidthMax;
extern float tLenMin;
extern float tLenMax;
extern float tAreaMax;
extern float tRatioMin;
extern float tRatioMax;
extern float minLenRatio;
extern float tPtPerM3;


class Cell{
private:
    float smoothed;
    float height;
    float hDiff;
    float hGround;
    float minZ;
    bool isGround;

public:
    Cell();
    void updateMinZ(float z);
    void updataHeight(float h) {height = h;}
    void updateSmoothed(float s) {smoothed = s;}
    void updateHDiff(float hd){hDiff = hd;}
    void updateGround(){isGround = true; hGround = height;}
    bool isThisGround(){return isGround;}
    float getMinZ() {return minZ;}
    float getHeight(){return height;}
    float getHDiff(){ return hDiff;}
    float getSmoothed() {return smoothed;}
    float getHGround() {return hGround;}
};



void createAndMapPolarGrid(pcl::PointCloud<pcl::PointXYZ> cloud,
                           std::array<std::array<Cell, numBin>, numChannel>& polarData );


void computeHDiffAdjacentCell(std::array<Cell, numBin>& channelData);

void groundRemove(pcl::PointCloud<pcl::PointXYZ>& cloud, 
                  pcl::PointCloud<pcl::PointXYZ>::Ptr elevatedCloud, 
                  pcl::PointCloud<pcl::PointXYZ>::Ptr groundCloud); 


/*
Functions for Guassian Smoothing
*/
double gauss(double sigma, double x);

std::vector<double> gaussKernel(int samples, double sigma);

void gaussSmoothen(std::array<Cell, numBin>& values, double sigma, int samples);

/*
Functions for Componenet Clustering
*/
void componentClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr elevatedCloud,
                         std::array<std::array<int, numGrid>, numGrid> & cartesianData,
                         int & numCluster);

void mapCartesianGrid(pcl::PointCloud<pcl::PointXYZ>::Ptr elevatedCloud,
                      std::array<std::array<int, numGrid>, numGrid> & cartesianData);

void makeClusteredCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& elevatedCloud,
                        std::array<std::array<int, numGrid>, numGrid> cartesianData,
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& clusterCloud);


/*
Functions for Box Fitting
*/
std::vector<pcl::PointCloud<pcl::PointXYZ>> boxFitting(pcl::PointCloud<pcl::PointXYZ>::Ptr elevatedCloud,
                        std::array<std::array<int, numGrid>, numGrid> cartesianData,
                        int numCluster);

visualization_msgs::Marker Draw_Bounding_Box(pcl::PointCloud<pcl::PointXYZ>& cluster_cloud,std_msgs::Header header,const int ID);

#endif //Cloud_Filter.hpp