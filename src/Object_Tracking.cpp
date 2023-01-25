
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <pcl_obstacle_detection/Object_Tracking.hpp>


ObjectTracker::ObjectTracker(float x_int,float y_int,double ini_time){
    // std::cout<< "Constructing ObjectTracker"<<std::endl;
    prev_time_ = ini_time;

    states_prev_ = Eigen::MatrixXd(4,1);
    states_prev_<< x_int,y_int,0.01,0.01;

    states_now_ = Eigen::MatrixXd(4,1); //initial condition
    states_now_ << x_int,y_int,0.01,0.01;
    
    P_prev_ = Eigen::MatrixXd(4,4);
    P_prev_<< 1e-6,0,0,0,
            0,1e-6,0,0,
            0,0,1e-6,0,
            0,0,0,1e-6;
    P_now_ = Eigen::MatrixXd(4,4);
    P_now_<< 1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1;
    
    R_ = Eigen::MatrixXd(4,4);
    R_ << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; //testing; no variance
    // R_<< 0.1,0,0,0,
    //      0,0.1,0,0,
    //      0,0,0.1,0,
    //      0,0,0,0.1;

    Q_ = Eigen::MatrixXd(2,2);
    Q_ << 0,0,0,0;

    // Q_<< 0.1,0,0,0.1;

    

    L = 4;
    alpha = 1e-3;
    ki = 0;
    beta = 2;
    lamda = pow(alpha,2) * (L+ki) - L;

    calculateWeight();

}

Eigen::MatrixXd ObjectTracker::sigmaPointSampling(Eigen::MatrixXd mean, Eigen::MatrixXd cov){

    Eigen::MatrixXd state_sigma(L,2*L+1);
    //Calculate Square Root of P
    Eigen::MatrixXd P_sqrt = cov.llt().matrixL();

    //populate the first sigma vector with states_
    state_sigma.col(0) = mean;

    //Calculate Sigma Points
    for (int i = 0;i<L;i++){
        state_sigma.col(1+i) = mean + sqrt(lamda + L) * P_sqrt.col(i);
        state_sigma.col(L+1+i) = mean - sqrt(lamda + L) * P_sqrt.col(i);
    }

    return state_sigma;

    // std::cout<< "state_sigma =" << state_sigma<< std::endl;
}

void ObjectTracker::calculateWeight(){
    //calculate weight for mean and covariance
    sigma_weight_mean = Eigen::MatrixXd(1,2*L+1);
    sigma_weight_cov  = Eigen::MatrixXd(1,2*L+1);

    sigma_weight_mean(0,0) = lamda /(L+lamda);
    sigma_weight_cov(0,0) = lamda /(L+lamda) + (1 - alpha*alpha + beta);

    for (int j = 1;j< (2*L+1) ;j++){
        sigma_weight_mean(0,j) = 1/(2*(lamda+L));
        sigma_weight_cov(0,j) = 1/(2*(lamda+L));
    }

    // std::cout<< "sigma_weight_mean =" << sigma_weight_mean<< std::endl;
    // std::cout<< "sigma_weight_cov =" << sigma_weight_cov<< std::endl;

}

void ObjectTracker::statePrediction(double delta_t){

    //Generate Sigma Points with previous states and covariances
    sigma_points_pre_ = sigmaPointSampling(states_prev_,P_prev_);

    // Propagate Sigma point through non-linear state predition
    sigma_points_mean_ = statePropagationCV(sigma_points_pre_,delta_t);

    //Predict the mean
    states_mean_ = Eigen::MatrixXd(L,1);
    
    for (int i = 0;i<sigma_points_mean_.rows();i++){
        states_mean_(i,0)= sigma_weight_mean.cwiseProduct(sigma_points_mean_.row(i)).sum();
        // std::cout<< "states_mean =" << states_mean_(i,0)<< std::endl;
    }
    
    std::cout<< "states_mean_ =" << states_mean_<< std::endl;

    //Predict the covariances
    P_mean_ = Eigen::MatrixXd(L,L);
    for (int i = 0;i<sigma_weight_cov.cols();i++){
        P_mean_+= sigma_weight_cov(0,i) * (sigma_points_mean_.col(i) - states_mean_)
                    *(sigma_points_mean_.col(i) - states_mean_).transpose();
    }
    
    P_mean_ += R_;


    std::cout<< "P_mean_=" << P_mean_<< std::endl;

    //Calculate New sets of Sigma_Points
    sigma_points_new_ = sigmaPointSampling(states_mean_,P_mean_);



    std::cout<< "sigma_points_new_=" << sigma_points_new_<< std::endl;


}

void ObjectTracker::stateUpdate(Eigen::MatrixXd measurement,double delta_t){
    // Measurement Update
    measurement_ = measurement;

    //Predict Observation Points
    measurement_pred_ = measurementPropagationCV(sigma_points_new_,delta_t);
    std::cout<< "measurement_pred_ =" << measurement_pred_<< std::endl;

    //Predicted measurement
    measurement_mean_ = Eigen::MatrixXd(measurement.rows(),measurement.cols());

    for (int i = 0; i< sigma_weight_mean.cols();i++){
        measurement_mean_ += sigma_weight_mean(0,i) * measurement_pred_.col(i);
    }
    std::cout<< "measurement_mean_ =" << measurement_mean_<< std::endl;

    //Innovation Covariances & cross_covariance
    Eigen::MatrixXd S(Q_.rows(),Q_.cols());
    Eigen::MatrixXd C_cov(states_mean_.rows(),measurement.rows());

    for (int i = 0; i< sigma_weight_cov.cols();i++){
        //Innovation Covariances
        S += sigma_weight_cov(0,i) * (measurement_pred_.col(i)-measurement_mean_) *
                    (measurement_pred_.col(i)-measurement_mean_).transpose();
        
        //Cross Covariances
        std::cout<< "sigma_point_col(i): "<< sigma_points_new_.col(i)<<std::endl;
        C_cov += sigma_weight_cov(0,i) * (sigma_points_new_.col(i)-states_mean_) *
                    (measurement_pred_.col(i)-measurement_mean_).transpose();
    }

    S += Q_;

    // Calculate Update K
    Eigen::MatrixXd K = C_cov * S.inverse();

    std::cout<< "K: "<< K<<std::endl;
    std::cout<< "state Gain: "<<K * (measurement - measurement_mean_)<<std::endl;

    //Find Corrected Mean
    states_now_ = states_mean_ + K * (measurement - measurement_mean_);

    //Find Corrected Covariances
    P_now_ = P_mean_ - K * S * K.transpose();


    // std::cout<< "states_now =" << states_now_<< std::endl;
    // std::cout<< "P_now_ =" << P_now_<< std::endl;
}


void ObjectTracker::UKFUpdate(Eigen::MatrixXd measurement,double time_now){
    if (time_now < 10){
        std::cout<< "delta_t=" << time_now << std::endl;
        statePrediction(time_now);
        stateUpdate(measurement,time_now); 
        // std::cout<<"position_prev_ before: ("<< states_prev_(0,0)<<", "<<states_prev_(1,0)<< ")"<<std::endl;
        states_prev_= states_now_;
        P_prev_ = P_now_;
        // std::cout<<"position_prev_ after: ("<< states_prev_(0,0)<<", "<<states_prev_(1,0)<< ")"<<std::endl;
        // prev_time_ = time_now;
        
        // std::cout<<"new prev_time_ - time_now:"<< prev_time_ - time_now <<std::endl;
        std::cout<<"new states: ("<< states_now_(0,0)<<", "<<states_now_(1,0)<< " ,"<<states_now_(3,0)<<" ,"<< states_now_(3,0) <<" )"<<std::endl;

    }
}

Eigen::MatrixXd ObjectTracker::statePropagationCV(Eigen::MatrixXd sigma_points,double delta_t){
    // Applied a Constant Velocity State Propagation

    Eigen::MatrixXd sigma_propagated(Eigen::MatrixXd(sigma_points.rows(),sigma_points.cols()));
    //Input should be (4*N) dimension
    for (int i = 0; i< sigma_points.cols();i++){
        sigma_propagated(0,i) = sigma_points(0,i) + sigma_points(2,i) * delta_t;//update x
        sigma_propagated(1,i) = sigma_points(1,i) + sigma_points(3,i) * delta_t;//update y
        sigma_propagated(2,i) = sigma_points(2,i); //update vel_x
        sigma_propagated(3,i) = sigma_points(3,i); //update vel_y
    }

    return sigma_propagated;


}

Eigen::MatrixXd ObjectTracker::measurementPropagationCV(Eigen::MatrixXd sigma_points,double delta_t){
    // Apply measurement update with x,y value

    Eigen::MatrixXd sigma_propagated(Eigen::MatrixXd(2,sigma_points.cols()));

    for (int i=0;i<sigma_points.cols();i++){
        sigma_propagated(0,i) = sigma_points(0,i);
        sigma_propagated(1,i) = sigma_points(1,i);
    }
    
    return sigma_propagated;
}