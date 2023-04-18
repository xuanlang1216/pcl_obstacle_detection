#ifndef MY_OBJECT_TRACKER_H
#define MY_OBJECT_TRACKER_H

#include <vector>
#include <string>
#include <fstream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
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

        int update_model_; //1:constant velocity model, 2: constant velocity model with shape(width,height)


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

        std::vector<double> x_history;
        std::vector<double> y_history;

        double last_measurement_x;
        double last_measurement_y;
        double last_measurement_width;
        double last_measurement_length;
        double last_measurement_time;
        // std::vector<double> z_history;

        
        // confident level of tracking
        double confident_level;
        int updated_count;



        //Covariances
        Eigen::MatrixXd P_prev_; //Previous Estimated State Covariance
        Eigen::MatrixXd P_mean_; //State Covariance after state propagation with sigma points
        Eigen::MatrixXd P_now_; //State Covariance after a complete UKF update

        Eigen::MatrixXd Q_; //Process Noise
        Eigen::MatrixXd R_; //Measurement Noise

        bool updated; // used to store if the object is update last frame

        int ID; // ID of the the Object for tracking
        
        // generate random color for the object
        double color_r_ ;
        double color_g_;
        double color_b_;

        /*
        Constructor
        */
        ObjectTracker(float x_initial,float y_initial,double time_initial,int ID_in);
        ObjectTracker(float x_initial,float y_initial,double time_initial,int ID_in,double width, double length);
        ObjectTracker(float x_initial,float y_initial,double time_initial,int ID_in,double width, double length,double acc);

        /*
        Destructor
        */
        // virtual ~ObjectTracker;

        Eigen::MatrixXd sigmaPointSampling(Eigen::MatrixXd mean, Eigen::MatrixXd cov);

        void calculateWeight();

        void statePrediction(double delta_t);

        void stateUpdate(Eigen::MatrixXd measurement,double delta_t);

        void UKFUpdate(Eigen::MatrixXd measurement,double time_now);

        //constant velocity model
        void statePropagateOnly(double time_now);
        Eigen::MatrixXd statePropagationCV(Eigen::MatrixXd sigma_points,double delta_t);
        Eigen::MatrixXd measurementPropagationCV(Eigen::MatrixXd sigma_points,double delta_t);

        //constant velocity model with width and length
        void statePropagateOnlywithShape(double time_now);
        Eigen::MatrixXd statePropagationCVwithShape(Eigen::MatrixXd sigma_points,double delta_t);
        Eigen::MatrixXd measurementPropagationCVwithShape(Eigen::MatrixXd sigma_points,double delta_t);
        
        // Constant accerlaration model
        Eigen::MatrixXd statePropagationCA(Eigen::MatrixXd sigma_points,double delta_t);
        Eigen::MatrixXd measurementPropagationCA(Eigen::MatrixXd sigma_points,double delta_t);
        void statePropagateOnlyCA(double time_now);

};


#endif //Object_Tracking.hpp