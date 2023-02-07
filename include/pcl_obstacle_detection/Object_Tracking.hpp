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
        Eigen::MatrixXd states_mean_; //states after state propagation
        Eigen::MatrixXd states_prev_; //previous states
        Eigen::MatrixXd states_now_;  //states after UKF update


        Eigen::MatrixXd measurement_mean_; // sum of all predicted measurement with Weight
        Eigen::MatrixXd measurement_;     // measurement in
        Eigen::MatrixXd measurement_pred_; //predicted measurement with sigma points

        Eigen::MatrixXd sigma_points_pre_; //Sigma points generated with previous states
        Eigen::MatrixXd sigma_points_mean_; //Sigma points after noise-free state prediction
        Eigen::MatrixXd sigma_points_new_; //new sets of Sigma points after state propagation


        int tracking_state; //1: just initilized, 2: updated last frame, 3: no update last frame, 4:lost tracked


        double prev_time_;
        double curr_time;


        int L; //Dimension of states_
        float lamda; //The scaling parameter

        float alpha; // Spead of the sigma points around states (usually small: ex. 1e-3)
        float ki;  //Secondary scaling parameter (usually 0)
        float beta; //incorporate prior knowledge of the distubation of states (2 for Gaussian distribution)

        Eigen::MatrixXd sigma_weight_mean;
        Eigen::MatrixXd sigma_weight_cov;


        Eigen::Matrix<float,5,5> F_;
        Eigen::Matrix<float,5,5> H_;


        //Covariances
        Eigen::MatrixXd P_prev_; //Previous Estimated State Covariance
        Eigen::MatrixXd P_mean_; //State Covariance after state propagation with sigma points
        Eigen::MatrixXd P_now_; //State Covariance after a complete UKF update

        Eigen::MatrixXd Q_; //Process Noise
        Eigen::MatrixXd R_; //Measurement Noise

        bool updated; // used to store if the object is update last frame

        int ID; // ID of the the Object for tracking

        /*
        Constructor
        */
        ObjectTracker(float x_int,float y_int,double ini_time);

        /*
        Destructor
        */
        // virtual ~ObjectTracker;

        Eigen::MatrixXd sigmaPointSampling(Eigen::MatrixXd mean, Eigen::MatrixXd cov);

        void calculateWeight();

        void statePrediction(double delta_t);

        void stateUpdate(Eigen::MatrixXd measurement,double delta_t);

        void UKFUpdate(Eigen::MatrixXd measurement,double time_now);

        void statePropagateOnly(double delta_t);


        Eigen::MatrixXd statePropagationCV(Eigen::MatrixXd sigma_points,double delta_t);

        Eigen::MatrixXd measurementPropagationCV(Eigen::MatrixXd sigma_points,double delta_t);

};


#endif //Object_Tracking.hpp