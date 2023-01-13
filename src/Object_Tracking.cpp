
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <pcl_obstacle_detection/Object_Tracking.hpp>


ObjectTracker::ObjectTracker(float x_int,float y_int){
    std::cout<< "Constructing ObjectTracker"<<std::endl;
    states_ = Eigen::MatrixXd(4,1);
    states_<< x_int,y_int,0,0;
    
    P_ = Eigen::MatrixXd(4,4);
    P_<< 1,0,0,0,
         0,6,0,0,
         0,0,11,0,
         0,0,0,16;
    

    L = 4;
    alpha = 1e-3;
    ki = 0.0;
    beta = 2;
    lamda = pow(alpha,2) * (L+ki) - L;

}

void ObjectTracker::sigmaPointSampling(){

    Eigen::MatrixXd state_sigma(L,2*L+1);
    //Calculate Square Root of P
    Eigen::MatrixXd P_sqrt = P_.llt().matrixL();

    //populate the first sigma vector with states_
    state_sigma.col(0) = states_;

    //Calculate Sigma Points
    for (int i = 0;i<L;i++){
        state_sigma.col(1+i) = states_ + sqrt(lamda + L) * P_sqrt.col(i);
        state_sigma.col(L+1+i) = states_ - sqrt(lamda + L) * P_sqrt.col(i);
    }

    std::cout<< "state_sigma =" << state_sigma<< std::endl;
    

}

void calculateWeight(){
    
}