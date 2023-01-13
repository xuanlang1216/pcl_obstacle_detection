#ifndef MY_OBJECT_TRACKER_H
#define MY_OBJECT_TRACKER_H

#include <vector>
#include <string>
#include <fstream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <vector>
#include <array>

class ObjectTracker{
    public:
        // [x,y,vel,yaw,yaw_rate]
        Eigen::MatrixXd states_;
        Eigen::MatrixXd measurement_;

        int L; //Dimension of states_
        float lamda; //The scaling parameter

        float alpha; // Spead of the sigma points around states (usually small: ex. 1e-3)
        float ki;  //Secondary scaling parameter (usually 0)
        float beta; //incorporate prior knowledge of the distubation of states (2 for Gaussian distribution)


        Eigen::Matrix<float,5,5> F_;
        Eigen::Matrix<float,5,5> H_;


        //Covariances
        Eigen::MatrixXd P_; //Estimated State Covariance
        Eigen::MatrixXd Q_; //Process Noise
        Eigen::MatrixXd R_; //Measurement Noise

        /*
        Constructor
        */
        ObjectTracker(float x_int,float y_int);

        /*
        Destructor
        */
        // virtual ~ObjectTracker;

        void sigmaPointSampling();

        void statePrediction(double delta_t);

        void state_Update();

};


#endif //Object_Tracking.hpp