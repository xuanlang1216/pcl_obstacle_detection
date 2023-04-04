

#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

#include <pcl_obstacle_detection/Cloud_Filter.hpp>

// Variables for Ground Removal
float rMin = 0.1;
float rMax = 120;
float tHmin = -2;
float tHmax = -0.5;
float tHDiff = 0.2;
float hSensor = 1.5; 

// Variables for Component_clustering
float roiM = 30; //distance to cluster
int kernelSize = 3;

// Variables for box fitting
float picScale = 900/roiM;
int ramPoints = 80;
int lSlopeDist = 3.0;
int lnumPoints = 300;
float sensorHeight = hSensor;
float tHeightMin = 1.0;  //Min height of the object
float tHeightMax = 2.6;  //max height of the object
float tWidthMin = 0.25;  //min Width of the object
float tWidthMax = 2;   //max wdith of the object
float tLenMin = 0.5;     //min length of the object
float tLenMax = 5.0;    //max length of the object
float tAreaMax = 5.0;   //max area of the object
float tRatioMin = 1.3;   //min ratio between length and width
float tRatioMax = 5.0;   //max ratio between length and width
float minLenRatio = 3.0; //min length of object for ratio check
float tPtPerM3 = 8;      //min point count per bouding box volume


Cell::Cell(){
    minZ = 1000;
    isGround = false;
}

void Cell::updateMinZ(float z) {
    if (z < minZ) minZ = z;
}

/*
Filter the cloud by distance to center
*/
void filterCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud, pcl::PointCloud<pcl::PointXYZ> & filteredCloud){
    for (int i = 0; i < cloud.size(); i++) {
        float x = cloud.points[i].x;
        float y = cloud.points[i].y;
        float z = cloud.points[i].z;

        float distance = sqrt(x * x + y * y);
        if(distance <= rMin || distance >= rMax) {
            continue; // filter out
        }
        else{
            pcl::PointXYZ o;
            o.x = x;
            o.y = y;
            o.z = z;
            filteredCloud.push_back(o);
        }
    }
}

/*
Calculate the cell index from x,y
*/
void getCellIndexFromPoints(float x, float y, int& chI, int& binI){
    float distance = sqrt(x * x + y * y);
    //normalize
    float chP = (atan2(y, x) + M_PI) / (2 * M_PI);
    float binP = (distance - rMin) / (rMax - rMin);
    //index
    chI = floor(chP*numChannel);
    binI = floor(binP*numBin);
//    cout << "bin ind: "<<binI << " ch ind: "<<chI <<endl;
}

void createAndMapPolarGrid(pcl::PointCloud<pcl::PointXYZ> cloud,
                           std::array<std::array<Cell, numBin>, numChannel>& polarData ){
    for (int i = 0; i < cloud.size(); i++) {
        float x = cloud.points[i].x;
        float y = cloud.points[i].y;
        float z = cloud.points[i].z;

        int chI, binI;
        getCellIndexFromPoints(x, y, chI, binI);
        // TODO; modify abobe function so that below code would not need
        if(chI < 0 || chI >=numChannel || binI < 0 || binI >= numBin) continue; // to prevent segentation fault
        polarData[chI][binI].updateMinZ(z);
    }
}

// update HDiff with larger value
void computeHDiffAdjacentCell(std::array<Cell, numBin>& channelData){
    for(int i = 0; i < channelData.size(); i++){
        // edge case
        if(i == 0){
            float hD = channelData[i].getHeight() - channelData[i+1].getHeight();
            channelData[i].updateHDiff(hD);
        }
        else if(i == channelData.size()-1){
            float hD = channelData[i].getHeight() - channelData[i-1].getHeight();
            channelData[i].updateHDiff(hD);
        }
        // non-edge case
        else{
            float preHD  = channelData[i].getHeight() - channelData[i-1].getHeight();
            float postHD = channelData[i].getHeight() - channelData[i+1].getHeight();
            if(preHD > postHD) channelData[i].updateHDiff(preHD);
            else channelData[i].updateHDiff(postHD);
        }

//        cout <<channelData[i].getHeight() <<" " <<channelData[i].getHDiff() << endl;
    }
}

void applyMedianFilter(std::array<std::array<Cell, numBin>, numChannel>& polarData){
    // maybe later: consider edge case
    for(int channel = 1; channel < polarData.size()-1; channel++){
        for(int bin = 1; bin < polarData[0].size()-1; bin++){
            if(!polarData[channel][bin].isThisGround()){
                // target cell is non-ground AND surrounded by ground cells
                if(polarData[channel][bin+1].isThisGround()&&
                   polarData[channel][bin-1].isThisGround()&&
                   polarData[channel+1][bin].isThisGround()&&
                   polarData[channel-1][bin].isThisGround()){
                    std::vector<float> sur{polarData[channel][bin+1].getHeight(),
                                      polarData[channel][bin-1].getHeight(),
                                      polarData[channel+1][bin].getHeight(),
                                      polarData[channel-1][bin].getHeight()};
                    sort(sur.begin(), sur.end());
                    float m1 = sur[1]; float m2 = sur[2];
                    float median = (m1+m2)/2;
                    polarData[channel][bin].updataHeight(median);
                    polarData[channel][bin].updateGround();
                }
            }
        }
    }
}


void outlierFilter(std::array<std::array<Cell, numBin>, numChannel>& polarData){
    for(int channel = 1; channel < polarData.size() - 1; channel++) {
        for (int bin = 1; bin < polarData[0].size() - 2; bin++) {
            if(polarData[channel][bin].isThisGround()&&
               polarData[channel][bin+1].isThisGround()&&
               polarData[channel][bin-1].isThisGround()&&
               polarData[channel][bin+2].isThisGround()){
                float height1 = polarData[channel][bin-1].getHeight();
                float height2 = polarData[channel][bin].getHeight();
                float height3 = polarData[channel][bin+1].getHeight();
                float height4 = polarData[channel][bin+2].getHeight();
                if(height1 != tHmin && height2 == tHmin && height3 != tHmin){
                    float newH = (height1 + height3)/2;
                    polarData[channel][bin].updataHeight(newH);
                    polarData[channel][bin].updateGround();
                }
                else if(height1 != tHmin && height2 == tHmin && height3 == tHmin && height4 != tHmin){
                    float newH = (height1 + height4)/2;
                    polarData[channel][bin].updataHeight(newH);
                    polarData[channel][bin].updateGround();
                }
            }
        }
    }
}


void groundRemove(pcl::PointCloud<pcl::PointXYZ> &  cloud,
              pcl::PointCloud<pcl::PointXYZ>::Ptr  elevatedCloud,
              pcl::PointCloud<pcl::PointXYZ>::Ptr  groundCloud){

    pcl::PointCloud<pcl::PointXYZ> filteredCloud;

    filterCloud(cloud, filteredCloud);
    std::array<std::array<Cell, numBin>, numChannel> polarData;
    createAndMapPolarGrid(filteredCloud, polarData);

    // cout << "size: "<<groundCloud->size() << endl;
    for (int channel = 0; channel < polarData.size(); channel++){
        for (int bin = 0; bin < polarData[0].size(); bin ++){
            float zi = polarData[channel][bin].getMinZ();
            if(zi > tHmin && zi < tHmax){polarData[channel][bin].updataHeight(zi);}
            else if(zi > tHmax){polarData[channel][bin].updataHeight(hSensor);}
            else {polarData[channel][bin].updataHeight(tHmin);}
        }
        //could replace gauss with gradient
//        computeGradientAdjacentCell(polarData[channel]);
        gaussSmoothen(polarData[channel], 1, 3);
//        std::cout << " finished smoothing at channel "<< channel << std::endl;
        computeHDiffAdjacentCell(polarData[channel]);

        for (int bin = 0; bin < polarData[0].size(); bin ++){
            if(polarData[channel][bin].getSmoothed() < tHmax &&
                    polarData[channel][bin].getHDiff() < tHDiff){
                polarData[channel][bin].updateGround();
            }
            else if(polarData[channel][bin].getHeight() < tHmax &&
                    polarData[channel][bin].getHDiff() < tHDiff){
                polarData[channel][bin].updateGround();
            }
        }
    }
    // implement MedianFilter
    applyMedianFilter(polarData);
    // smoothen spot with outlier
    outlierFilter(polarData);

    for(int i = 0; i < filteredCloud.size(); i++) {
        float x = filteredCloud.points[i].x;
        float y = filteredCloud.points[i].y;
        float z = filteredCloud.points[i].z;

        pcl::PointXYZ o;
        o.x = x;
        o.y = y;
        o.z = z;
        int chI, binI;
        getCellIndexFromPoints(x, y, chI, binI);
        // assert(chI < 0 || chI >=numChannel || binI < 0 || binI >= numBin);
        if(chI < 0 || chI >=numChannel || binI < 0 || binI >= numBin) continue;
        
        if (polarData[chI][binI].isThisGround()) {
            float hGround = polarData[chI][binI].getHGround();
            if (z < (hGround + 0.25)) {
                groundCloud->push_back(o);
            } else {
                elevatedCloud->push_back(o);
            }
        } else {
            elevatedCloud->push_back(o);
        }
    }
}


/*
Guassian Smoothing
*/
double gauss(double sigma, double x) {
    double expVal = -1 * (pow(x, 2) / pow(2 * sigma, 2));
    double divider = sqrt(2 * M_PI * pow(sigma, 2));
    return (1 / divider) * exp(expVal);
}

std::vector<double> gaussKernel(int samples, double sigma) {
    std::vector<double> kernel(samples);
    double mean = samples/2;
    double sum = 0.0; // For accumulating the kernel values
    for (int x = 0; x < samples; ++x) {
        kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0)))/(2 * M_PI * sigma * sigma);
        // Accumulate the kernel values
        sum += kernel[x];
    }

// Normalize the kernel
    for (int x = 0; x < samples; ++x){
        kernel[x] /= sum;
    }

    // std::cout << "The kernel contains " << kernel.size() << " entries:";
    for (auto it = kernel.begin(); it != kernel.end(); ++it) {
        // std::cout << ' ' << *it;
    }
    // std::cout << std::endl;
    assert(kernel.size() == samples);

    return kernel;
}

void gaussSmoothen(std::array<Cell, numBin>& values, double sigma, int samples) {
    auto kernel = gaussKernel(samples, sigma);
    int sampleSide = samples / 2;
    unsigned long ubound = values.size();
    // applying gaussian kernel with zero padding
    for (long i = 0; i < ubound; i++) {
        double smoothed = 0;
        for (long j = i - sampleSide; j <= i + sampleSide; j++) {
            if (j >= 0 && j < ubound) {
                int sampleWeightIndex = sampleSide + (j - i);
                smoothed += kernel[sampleWeightIndex] * values[j].getHeight();
            }
        }
        // std::cout << " V: " << values[i].getHeight() << " SM: " << smoothed << std::endl;
        values[i].updateSmoothed(smoothed);
    }
}


/*
Component Clustering
*/
void mapCartesianGrid(pcl::PointCloud<pcl::PointXYZ>::Ptr elevatedCloud,
                        std::array<std::array<int, numGrid>, numGrid> & cartesianData){

    for(int i = 0; i < elevatedCloud->size(); i++){
        float x = elevatedCloud->points[i].x;
        float y = elevatedCloud->points[i].y;
        float xC = x+roiM/2; 
        float yC = y+roiM/2;
        // exclude outside roi points
        if(xC < 0 || xC >= roiM || yC < 0 || yC >=roiM) continue; //if the point is not within roiM, ignore it
        int xI = floor(numGrid*xC/roiM);
        int yI = floor(numGrid*yC/roiM);
        cartesianData[xI][yI] = -1; //if there is a point in that grid, mark that grid as occupied
//        int a = 0;
    }
}

void search(std::array<std::array<int, numGrid>, numGrid> & cartesianData, int clusterId, int cellX, int cellY){
    cartesianData[cellX][cellY] = clusterId;
    int mean = kernelSize/2;
    for (int kX = 0; kX < kernelSize; kX++){
        int kXI = kX-mean;
        if((cellX + kXI) < 0 || (cellX + kXI) >= numGrid) continue; //
        for( int kY = 0; kY < kernelSize;kY++){
            int kYI = kY-mean;
            if((cellY + kYI) < 0 || (cellY + kYI) >= numGrid) continue;

            if(cartesianData[cellX + kXI][cellY + kYI] == -1){
                search(cartesianData, clusterId, cellX +kXI, cellY + kYI);
            }

        }
    }
}

void findComponent(std::array<std::array<int, numGrid>, numGrid> & cartesianData, int &clusterId){
    // for each occupied cell, if the cell in the kernel is also occupied, mark it with the same clusterID.
    for(int cellX = 0; cellX < numGrid; cellX++){
        for(int cellY = 0; cellY < numGrid; cellY++){
            if(cartesianData[cellX][cellY] == -1){
                clusterId ++;
                search(cartesianData, clusterId, cellX, cellY);
            }
        }
    }
}

void componentClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr elevatedCloud,
                         std::array<std::array<int, numGrid>, numGrid> & cartesianData,
                         int & numCluster){
    // map 120m radius data(polar grid data) into 100x100 cartesian grid,
    // parameter might need to be modified
    // in this case 30mx30m with 100x100x grid
    mapCartesianGrid(elevatedCloud, cartesianData);
    findComponent(cartesianData, numCluster);
}


void makeClusteredCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& elevatedCloud,
                        std::array<std::array<int, numGrid>, numGrid> cartesianData,
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& clusterCloud){

    // cartesian data:: occupancy grid data

    for(int i = 0; i < elevatedCloud->size(); i++){
        float x = elevatedCloud->points[i].x;
        float y = elevatedCloud->points[i].y;
        float z = elevatedCloud->points[i].z;
        float xC = x+roiM/2;
        float yC = y+roiM/2;
        // exclude outside roi points
        if(xC < 0 || xC >= roiM || yC < 0 || yC >=roiM) continue;
        int xI = floor(numGrid*xC/roiM);
        int yI = floor(numGrid*yC/roiM);

        int clusterNum = cartesianData[xI][yI];
        if(clusterNum != 0){
            pcl::PointXYZRGB o;
            o.x = x;
            o.y = y;
            o.z = z;
            o.r = (500*clusterNum)%255;
            o.g = (100*clusterNum)%255;
            o.b = (150*clusterNum)%255;
            clusterCloud->push_back(o);
        }
    }
}

/*
Box Fitting
*/
void getClusteredPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr elevatedCloud,
                        std::array<std::array<int, numGrid>, numGrid> cartesianData,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>>&  clusteredPoints) {
    for (int i = 0; i < elevatedCloud->size(); i++) {
        float x = elevatedCloud->points[i].x;
        float y = elevatedCloud->points[i].y;
        float z = elevatedCloud->points[i].z;
        float xC = x + roiM / 2;
        float yC = y + roiM / 2;
        // exclude outside roi points
        if (xC < 0 || xC >= roiM || yC < 0 || yC >= roiM) continue;
        int xI = floor(numGrid * xC / roiM);
        int yI = floor(numGrid * yC / roiM);

        int clusterNum = cartesianData[xI][yI]; //1 ~ numCluster
        int vectorInd = clusterNum - 1; //0 ~ (numCluster -1 )
        if (clusterNum != 0) {
            pcl::PointXYZ o;
            o.x = x;
            o.y = y;
            o.z = z;
            clusteredPoints[vectorInd].push_back(o);
        }
    }
}

void getPointsInPcFrame(cv::Point2f rectPoints[], std::vector<cv::Point2f>& pcPoints, int offsetX, int offsetY){
    // loop 4 rect points
    for (int pointI = 0; pointI < 4; pointI++){
        float picX = rectPoints[pointI].x;
        float picY = rectPoints[pointI].y;
        // reverse offset
        float rOffsetX = picX - offsetX;
        float rOffsetY = picY - offsetY;
        // reverse from image coordinate to eucledian coordinate
        float rX = rOffsetX;
        float rY = picScale*roiM - rOffsetY;
        // reverse to 30mx30m scale
        float rmX = rX/picScale;
        float rmY = rY/picScale;
        // reverse from (0 < x,y < 30) to (-15 < x,y < 15)
        float pcX = rmX - roiM/2;
        float pcY = rmY - roiM/2;
        cv::Point2f point(pcX, pcY);
        pcPoints[pointI] = point;
    }
}

bool ruleBasedFilter(std::vector<cv::Point2f> pcPoints, float maxZ, int numPoints){
    bool isPromising = false;
    //minnimam points thresh
    if(numPoints < 25) 
    {
        // std::cout<<"numPoints not promising: "<< numPoints<< std::endl;
        return isPromising;
    }
    // length is longest side of the rectangle while width is the shorter side.
    float width, length, height, area, ratio, mass;

    float x1 = pcPoints[0].x;
    float y1 = pcPoints[0].y;
    float x2 = pcPoints[1].x;
    float y2 = pcPoints[1].y;
    float x3 = pcPoints[2].x;
    float y3 = pcPoints[2].y;

    float dist1 = sqrt((x1-x2)*(x1-x2)+ (y1-y2)*(y1-y2));
    float dist2 = sqrt((x3-x2)*(x3-x2)+ (y3-y2)*(y3-y2));
    if(dist1 > dist2){
        length = dist1;
        width = dist2;
    }
    else{
        length = dist2;
        width = dist1;
    }
    // assuming ground = sensor height
    height = maxZ + sensorHeight;

    //testing the number of point:
    // std::cout<<"# of point is: "<<numPoints<<std::endl;
    // std::cout<<"object shape: ("<<height<<","<<width<<","<<length<<","<<area<<","<<ratio<<","<<mass<<")"<<std::endl;


    // assuming right angle
    area = dist1*dist2;
    mass = area*height;
    ratio = length/width;

    //start rule based filtering
    if(height > tHeightMin && height < tHeightMax){
        if(width > tWidthMin && width < tWidthMax){
            if(length > tLenMin && length < tLenMax){
                if(area < tAreaMax){
                    if(numPoints > mass*tPtPerM3){
                        if(length > minLenRatio){
                            if(ratio > tRatioMin && ratio < tRatioMax){
                                isPromising = true;
                                return isPromising;
                            }
                        }
                        else{
                            isPromising = true;
                            return isPromising;
                        }
                    }
                }
            }
        }
    }
    else 
    {
        return isPromising;
    }
}



void getBoundingBox(std::vector<pcl::PointCloud<pcl::PointXYZ>>  clusteredPoints,
                    std::vector<pcl::PointCloud<pcl::PointXYZ>>& bbPoints){
    
    for (int iCluster = 0; iCluster < clusteredPoints.size(); iCluster++){
        // std::cout<< "processing cluster "<<iCluster<<std::endl;
        cv::Mat m (picScale*roiM, picScale*roiM, CV_8UC1, cv::Scalar(0));
        float initPX = clusteredPoints[iCluster][0].x + roiM/2;
        float initPY = clusteredPoints[iCluster][0].y + roiM/2;
        int initX = floor(initPX*picScale);
        int initY = floor(initPY*picScale);
        int initPicX = initX;
        int initPicY = picScale*roiM - initY;
        int offsetInitX = roiM*picScale/2 - initPicX;
        int offsetInitY = roiM*picScale/2 - initPicY;

        int numPoints = clusteredPoints[iCluster].size();
        std::vector<cv::Point> pointVec(numPoints);
        std::vector<cv::Point2f> pcPoints(4);
        float minMx, minMy, maxMx, maxMy;
        float minM = 999; float maxM = -999; float maxZ = -99;
        // for center of gravity
        float sumX = 0; float sumY = 0;
        for (int iPoint = 0; iPoint < clusteredPoints[iCluster].size(); iPoint++){
            float pX = clusteredPoints[iCluster][iPoint].x;
            float pY = clusteredPoints[iCluster][iPoint].y;
            float pZ = clusteredPoints[iCluster][iPoint].z;
            // cast (-15 < x,y < 15) into (0 < x,y < 30)
            float roiX = pX + roiM/2;
            float roiY = pY + roiM/2;
            // cast 30mx30m into 900x900 scale
            int x = floor(roiX*picScale);
            int y = floor(roiY*picScale);
            // cast into image coordinate
            int picX = x;
            int picY = picScale*roiM - y;
            // offset so that the object would be locate at the center
            int offsetX = picX + offsetInitX;
            int offsetY = picY + offsetInitY;
            m.at<uchar>(offsetY, offsetX) = 255;
            pointVec[iPoint] = cv::Point(offsetX, offsetY);
            // calculate min and max slope
            float m = pY/pX;
            if(m < minM) {
                minM = m;
                minMx = pX;
                minMy = pY;
            }
            if(m > maxM) {
                maxM = m;
                maxMx = pX;
                maxMy = pY;
            }

            //get maxZ
            if(pZ > maxZ) maxZ = pZ;

            sumX += offsetX;
            sumY += offsetY; 
        }
        // L shape fitting parameters
        float xDist = maxMx - minMx;
        float yDist = maxMy - minMy;
        float slopeDist = sqrt(xDist*xDist + yDist*yDist);
        float slope = (maxMy - minMy)/(maxMx - minMx);

        // random variable
        std::mt19937_64 mt(0);
        std::uniform_int_distribution<> randPoints(0, numPoints-1);

        // start l shape fitting for car like object
        // lSlopeDist = 30, lnumPoints = 300
        if(slopeDist > lSlopeDist && numPoints > lnumPoints)
        {
            float maxDist = 0;
            float maxDx, maxDy;

            // 80 random points, get max distance
            for(int i = 0; i < ramPoints; i++){
                int pInd = randPoints(mt);
                assert(pInd >= 0 && pInd < clusteredPoints[iCluster].size());
                float xI = clusteredPoints[iCluster][pInd].x;
                float yI = clusteredPoints[iCluster][pInd].y;

                // from equation of distance between line and point
                float dist = abs(slope*xI-1*yI+maxMy-slope*maxMx)/sqrt(slope*slope + 1);
                if(dist > maxDist) {
                    maxDist = dist;
                    maxDx = xI;
                    maxDy = yI;
                }
            }

            // for center of gravity
            // maxDx = sumX/clusteredPoints[iCluster].size();
            // maxDy = sumY/clusteredPoints[iCluster].size();

            // vector adding
            float maxMvecX = maxMx - maxDx;
            float maxMvecY = maxMy - maxDy;
            float minMvecX = minMx - maxDx;
            float minMvecY = minMy - maxDy;
            float lastX = maxDx + maxMvecX + minMvecX;
            float lastY = maxDy + maxMvecY + minMvecY;

            pcPoints[0] = cv::Point2f(minMx, minMy);
            pcPoints[1] = cv::Point2f(maxDx, maxDy);
            pcPoints[2] = cv::Point2f(maxMx, maxMy);
            pcPoints[3] = cv::Point2f(lastX, lastY);

            bool isPromising = ruleBasedFilter(pcPoints, maxZ, numPoints);
            if(!isPromising) continue;
        }
        else{
            //MAR fitting
            cv::RotatedRect rectInfo = cv::minAreaRect(pointVec);
            cv::Point2f rectPoints[4]; rectInfo.points( rectPoints );
            // covert points back to lidar coordinate
            getPointsInPcFrame(rectPoints, pcPoints, offsetInitX, offsetInitY);
            // rule based filter
            bool isPromising = ruleBasedFilter(pcPoints, maxZ, numPoints);
            if(!isPromising) continue;
        }

        // make pcl cloud for 3d bounding box
        pcl::PointCloud<pcl::PointXYZ> oneBbox;
        for(int pclH = 0; pclH < 2; pclH++){
            for(int pclP = 0; pclP < 4; pclP++){
                pcl::PointXYZ o;
                o.x = pcPoints[pclP].x;
                o.y = pcPoints[pclP].y;
                if(pclH == 0) 
                {
                    o.z = -sensorHeight;
                }
                else 
                {
                    o.z = maxZ;
                }
                oneBbox.push_back(o);
            }
        }
        bbPoints.push_back(oneBbox);
//        clustered2D[iCluster] = m;
    }

}

std::vector<pcl::PointCloud<pcl::PointXYZ>> boxFitting(pcl::PointCloud<pcl::PointXYZ>::Ptr elevatedCloud,
                std::array<std::array<int, numGrid>, numGrid> cartesianData,
                int numCluster){
    std::vector<pcl::PointCloud<pcl::PointXYZ>>  clusteredPoints(numCluster);
    getClusteredPoints(elevatedCloud, cartesianData, clusteredPoints);
    std::vector<pcl::PointCloud<pcl::PointXYZ>>  bbPoints;
    getBoundingBox(clusteredPoints, bbPoints);
    // std::cout<<"Find bbpoints: "<<bbPoints.size()<<std::endl;
    // std::cout<<"Find clusterPoints: "<<clusteredPoints.size()<<std::endl;
    return bbPoints; //use rule based filter
    // return clusteredPoints; // don't use the rule based filter
//    vector<vector<float>>  bBoxes(numCluster,  vector<float>(6));
//
//    return bBoxes;
}

visualization_msgs::Marker Draw_Bounding_Box(pcl::PointCloud<pcl::PointXYZ>& cluster_cloud,std_msgs::Header header,const int ID)
{
    Eigen::Vector4f centroid;
    Eigen::Vector4f min;
    Eigen::Vector4f max;

    pcl::compute3DCentroid(cluster_cloud,centroid);
    pcl::getMinMax3D(cluster_cloud,min,max);

    uint32_t shape = visualization_msgs::Marker::CUBE;
    visualization_msgs::Marker marker;
    

    marker.ns = "cube";
    marker.id = ID;
    marker.type = shape;
    marker.action = visualization_msgs::Marker::ADD;

    marker.header = header;

    marker.pose.position.x = centroid[0];
    marker.pose.position.y = centroid[1];
    marker.pose.position.z = centroid[2];
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = (max[0]-min[0]); //width
    marker.scale.y = (max[1]-min[1]); //length
    marker.scale.z = (max[2]-min[2]); //height

    marker.color.g = 1.0f;
    marker.color.b = 1.0;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration(5);

    return marker;   
}


/*
Tracking
*/

