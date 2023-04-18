
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <pcl_obstacle_detection/Object_Tracking.hpp>


ObjectTracker::ObjectTracker(float x_initial,float y_initial,double time_initial,int ID_in){
    update_model_ = 1;
    // std::cout<< "Constructing ObjectTracker"<<std::endl;
    tracking_state = 1; // just initialized
    updated = false;
    confident_level = 1.0;
    updated_count=1;

    prev_time_ = time_initial;

    states_prev_ = Eigen::MatrixXd(4,1);
    states_prev_<< x_initial,y_initial,0.0,0.0;

    states_now_ = Eigen::MatrixXd(4,1); //initial condition
    states_now_ << x_initial,y_initial,0.0,0.0;
    
    P_prev_ = Eigen::MatrixXd(4,4);
    P_prev_<< 1e-6,0,0,0,
            0,1e-6,0,0,
            0,0,1e-6,0,
            0,0,0,1e-6;
    P_now_ = Eigen::MatrixXd(4,4);
    P_now_<< 1e-6,0,0,0,
            0,1e-6,0,0,
            0,0,1e-6,0,
            0,0,0,1e-6;
    
    R_ = Eigen::MatrixXd(4,4);
    // R_ << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; //testing; no variance
    R_<< 0.1,0,0,0,
         0,0.1,0,0,
         0,0,0.1,0,
         0,0,0,0.1;

    Q_ = Eigen::MatrixXd(2,2);
    // Q_ << 0,0,0,0;

    Q_<< 0.1,0,0,0.1;

    ID = ID_in;

    L = 4;
    alpha = 1e-3;
    ki = 0;
    beta = 2;
    lamda = pow(alpha,2) * (L+ki) - L;


    // update history:
    x_history.push_back(states_now_(0,0));
    y_history.push_back(states_now_(1,0));

    last_measurement_x = x_initial;
    last_measurement_y = y_initial;
    last_measurement_width =  0.0;
    last_measurement_length =  0.0;  
    last_measurement_time = time_initial;

    // initialize object color:
    color_r_ = ((500*ID)%255)/255.0;
    color_g_ = ((100*ID)%255)/255.0;
    color_b_ = ((150*ID)%255)/255.0;

    calculateWeight();
}

ObjectTracker::ObjectTracker(float x_initial,float y_initial,double time_initial,int ID_in,double width, double length){
    // std::cout<< "Constructing ObjectTracker with width and length"<<std::endl;

    update_model_ = 2;
    tracking_state = 1; // just initialized
    updated = false;
    confident_level = 1.0;
    updated_count=1;

    prev_time_ = time_initial;

    states_prev_ = Eigen::MatrixXd(6,1);
    states_prev_<< x_initial,y_initial,0.0,0.0,width,length;

    states_now_ = Eigen::MatrixXd(6,1); //initial condition
    states_now_ << x_initial,y_initial,0.0,0.0,width,length;
    
    P_prev_ = Eigen::MatrixXd(6,6);
    P_prev_<< 1e-6,0,0,0,0,0,
            0,1e-6,0,0,0,0,
            0,0,1e-6,0,0,0,
            0,0,0,1e-6,0,0,
            0,0,0,0,1e-6,0,
            0,0,0,0,0,1e-6;
    P_now_ = Eigen::MatrixXd(6,6);
    P_now_<< 1e-6,0,0,0,0,0,
            0,1e-6,0,0,0,0,
            0,0,1e-6,0,0,0,
            0,0,0,1e-6,0,0,
            0,0,0,0,1e-6,0,
            0,0,0,0,0,1e-6;
    
    R_ = Eigen::MatrixXd(6,6);
    // R_ << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; //testing; no variance
    R_<< 0.01,0,0,0,0,0,
         0,0.01,0,0,0,0,
         0,0,0.01,0,0,0,
         0,0,0,0.01,0,0,
         0,0,0,0,0.01,0,
         0,0,0,0,0,0.01;

    Q_ = Eigen::MatrixXd(4,4);
    // Q_ << 0,0,0,0;

    Q_<< 0.01,0,0,0,
        0,0.01,0,0,
        0,0,0.01,0,
        0,0,0,0.01;

    ID = ID_in;

    L = 6;
    alpha = 1e-3;
    ki = 0;
    beta = 2;
    lamda = pow(alpha,2) * (L+ki) - L;


    // update history:
    x_history.push_back(states_now_(0,0));
    y_history.push_back(states_now_(1,0));

    last_measurement_x = x_initial;
    last_measurement_y = y_initial;
    last_measurement_width =  width;
    last_measurement_length =  length;  
    last_measurement_time = time_initial;

    // initialize object color:
    color_r_ = ((500*ID)%255)/255.0;
    color_g_ = ((100*ID)%255)/255.0;
    color_b_ = ((150*ID)%255)/255.0;

    calculateWeight();

}

ObjectTracker::ObjectTracker(float x_initial,float y_initial,double time_initial,int ID_in,double width, double length,double acc){
    // std::cout<< "Constructing ObjectTracker with Constant Acceleration Model"<<std::endl;

    update_model_ = 3;
    tracking_state = 1; // just initialized
    updated = false;
    confident_level = 1.0;
    updated_count=1;

    prev_time_ = time_initial;

    // x,y,x_vel,y_vel,width,length,x_acc,y_acc
    states_prev_ = Eigen::MatrixXd(8,1);
    states_prev_<< x_initial,y_initial,0.0,0.0,width,length,0.0,0.0;

    states_now_ = Eigen::MatrixXd(8,1); //initial condition
    states_now_ << x_initial,y_initial,0.0,0.0,width,length,0.0,0.0;
    
    // P_prev_ = Eigen::MatrixXd(8,8);
    // P_prev_<< 1e-6,0,0,0,0,0,
    //         0,1e-6,0,0,0,0,
    //         0,0,1e-6,0,0,0,
    //         0,0,0,1e-6,0,0,
    //         0,0,0,0,1e-6,0,
    //         0,0,0,0,0,1e-6;
    P_prev_ = Eigen::MatrixXd::Identity(8,8) * 1e-6;

    // P_now_ = Eigen::MatrixXd(6,6);
    // P_now_<< 1e-6,0,0,0,0,0,
    //         0,1e-6,0,0,0,0,
    //         0,0,1e-6,0,0,0,
    //         0,0,0,1e-6,0,0,
    //         0,0,0,0,1e-6,0,
    //         0,0,0,0,0,1e-6;
    P_now_ = Eigen::MatrixXd::Identity(8,8) * 1e-6;
    
    // R_ = Eigen::MatrixXd(6,6);
    // // R_ << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; //testing; no variance
    // R_<< 0.01,0,0,0,0,0,
    //      0,0.01,0,0,0,0,
    //      0,0,0.01,0,0,0,
    //      0,0,0,0.01,0,0,
    //      0,0,0,0,0.01,0,
    //      0,0,0,0,0,0.01;
    R_ = Eigen::MatrixXd::Identity(8,8) * 0.01;

    Q_ = Eigen::MatrixXd(4,4);

    Q_<< 0.01,0,0,0,
        0,0.01,0,0,
        0,0,0.01,0,
        0,0,0,0.01;

    ID = ID_in;

    L = 8;
    alpha = 1e-3;
    ki = 0;
    beta = 2;
    lamda = pow(alpha,2) * (L+ki) - L;


    // update history:
    x_history.push_back(states_now_(0,0));
    y_history.push_back(states_now_(1,0));

    last_measurement_x = x_initial;
    last_measurement_y = y_initial;
    last_measurement_width =  width;
    last_measurement_length =  length; 
    last_measurement_time = time_initial;

    // initialize object color:
    color_r_ = ((500*ID)%255)/255.0;
    color_g_ = ((100*ID)%255)/255.0;
    color_b_ = ((150*ID)%255)/255.0;

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
    if (update_model_ == 1){
        sigma_points_mean_ = statePropagationCV(sigma_points_pre_,delta_t);
    }
    else if(update_model_ == 2)
    {
        sigma_points_mean_ = statePropagationCVwithShape(sigma_points_pre_,delta_t);
    }
    else if (update_model_ == 3){
        sigma_points_mean_ = statePropagationCA(sigma_points_pre_,delta_t);
    }
    else{
        sigma_points_mean_ = sigma_points_pre_;
        std::cout<< "Wrong update model!!!!!!!"<<std::endl;
    }


    //Predict the mean
    states_mean_ = Eigen::MatrixXd(L,1);
    
    for (int i = 0;i<sigma_points_mean_.rows();i++){
        states_mean_(i,0)= sigma_weight_mean.cwiseProduct(sigma_points_mean_.row(i)).sum();
        // std::cout<< "states_mean =" << states_mean_(i,0)<< std::endl;
    }
    
    // std::cout<< "states_mean_ =" << states_mean_<< std::endl;

    //Predict the covariances
    P_mean_ = Eigen::MatrixXd(L,L);
    for (int i = 0;i<sigma_weight_cov.cols();i++){
        P_mean_+= sigma_weight_cov(0,i) * (sigma_points_mean_.col(i) - states_mean_)
                    *(sigma_points_mean_.col(i) - states_mean_).transpose();
    }
    
    P_mean_ += R_;


    // std::cout<< "P_mean_=" << P_mean_<< std::endl;

    //Calculate New sets of Sigma_Points
    sigma_points_new_ = sigmaPointSampling(states_mean_,P_mean_);



    // std::cout<< "sigma_points_new_=" << sigma_points_new_<< std::endl;


}

void ObjectTracker::stateUpdate(Eigen::MatrixXd measurement,double delta_t){
    // Measurement Update
    measurement_ = measurement;

    //Predict Observation Points
    // std::cout<<"Update_model"<<update_model_<<std::endl;
    if (update_model_ == 1){
        measurement_pred_ = measurementPropagationCV(sigma_points_new_,delta_t);
    }
    else if (update_model_ == 2){
        measurement_pred_ = measurementPropagationCVwithShape(sigma_points_new_,delta_t);
    }
    else if (update_model_ == 3){
        measurement_pred_ = measurementPropagationCA(sigma_points_new_,delta_t);
    }
    else{
        std::cout<<"WRONG UPDATE MODEL!!!!"<<std::endl;
    }


    //Predicted measurement
    measurement_mean_ = Eigen::MatrixXd::Zero(measurement_.rows(),measurement_.cols());

    for (int i = 0; i< sigma_weight_mean.cols();i++){
        measurement_mean_ += sigma_weight_mean(0,i) * measurement_pred_.col(i);
        // std::cout<<"sigma_weight_mean(0,i) * measurement_pred_.col(i);"<<std::endl;
        // std::cout<<sigma_weight_mean(0,i) * measurement_pred_.col(i)<<std::endl;
        // std::cout<<"measurement_mean_"<<std::endl;
        // std::cout<<measurement_mean_<<std::endl;
    }

    //Innovation Covariances & cross_covariance
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(Q_.rows(),Q_.cols());
    Eigen::MatrixXd C_cov = Eigen::MatrixXd::Zero(states_mean_.rows(),measurement_.rows());

    // std::cout<< "sigma_weight_mean" <<std::endl;
    // std::cout<<sigma_weight_mean<< std::endl;    


    // std::cout<< "measurement_mean_ =" <<std::endl;
    // std::cout<< measurement_mean_<< std::endl;

    // std::cout<< "S:"<<std::endl;
    // std::cout<< S<<std::endl;

    // std::cout<< "states_mean_.rows()"<<std::endl;
    // std::cout<< states_mean_.rows()<<std::endl;

    // std::cout<< "measurement_.rows()"<<std::endl;
    // std::cout<< measurement_.rows()<<std::endl;

    // std::cout<< "C_cov initial:"<<std::endl;
    // std::cout<<C_cov<<std::endl;

    // std::cout<<"measurement_pred_:"<<std::endl;
    // std::cout<<measurement_pred_<<std::endl;

    // std::cout<<"sigma_points_new_:"<<std::endl;
    // std::cout<<sigma_points_new_<<std::endl;

    // std::cout<<"states_mean_:"<<std::endl;
    // std::cout<< states_mean_<<std::endl;
    
    // std::cout<<"sigma_weight_cov: "<<std::endl;
    // std::cout<<sigma_weight_cov<<std::endl;

    for (int i = 0; i< sigma_weight_cov.cols();i++){
        //Innovation Covariances
        S += sigma_weight_cov(0,i) * (measurement_pred_.col(i)-measurement_mean_) *
                    (measurement_pred_.col(i)-measurement_mean_).transpose();
        
        //Cross Covariances
        // std::cout<< "C_cov "<<i<<": "<<std::endl;
        // std::cout<<C_cov<<std::endl;

        C_cov += sigma_weight_cov(0,i) * (sigma_points_new_.col(i)-states_mean_) *
                    (measurement_pred_.col(i)-measurement_mean_).transpose();
    }

    S += Q_;

    // Calculate Update K
    Eigen::MatrixXd K = C_cov * S.inverse();

    // std::cout<< "C_cov: "<< C_cov<<std::endl;
    // std::cout<<"S_inverse"<<S.inverse()<<std::endl;


    //Find Corrected Mean
    states_now_ = states_mean_ + K * (measurement_- measurement_mean_);

    //Find Corrected Covariances
    P_now_ = P_mean_ - K * S * K.transpose();

    // if (ID = 2){
    //     std::cout<< "states_mean_ =" << states_mean_<< std::endl;
    //     std::cout<< "states_now =" << states_now_<< std::endl;

    //     std::cout<< "K: "<< K<<std::endl;
    //     std::cout<<"measurement_"<< measurement_ <<std::endl;
    //     std::cout<<"measurement_pred_"<< measurement_pred_ <<std::endl;
    //     std::cout<<"measurement_mean"<< measurement_mean_ <<std::endl;
    //     std::cout<<"Measurement DIfference"<<(measurement_ - measurement_mean_)<<std::endl;
    //     std::cout<< "state Gain: "<<K * (measurement_ - measurement_mean_)<<std::endl;
    //     // std::cout<< "P_now_ =" << P_now_<< std::endl;
    // }
}


void ObjectTracker::UKFUpdate(Eigen::MatrixXd measurement,double time_now){
    double time_diff = time_now - prev_time_;

    if (time_diff < 10){
        // std::cout<< "delta_t=" << time_now << std::endl;
        // std::cout<< "Measurement: ("<< measurement(0,0)<<" ,"<<measurement(1,0)<<")"<<std::endl;
        // std::cout<<"states before update: ("<< states_now_(0,0)<<", "<<states_now_(1,0)<<", "<<states_now_(2,0)<<", "<<states_now_(3,0)<< ")"<<std::endl;
        // std::cout<<"Covariance before update"<< P_now_<<std::endl;
        statePrediction(time_diff);
        stateUpdate(measurement,time_diff); 
        // std::cout<<"states after update: ("<< states_now_(0,0)<<", "<<states_now_(1,0)<<", "<<states_now_(2,0)<<", "<<states_now_(3,0)<< ")"<<std::endl;
        // std::cout<<"Covariance after update"<< P_now_<<std::endl;
        states_prev_= states_now_;
        P_prev_ = P_now_;

        tracking_state = 2; // got updated
        updated = true;
        prev_time_ = time_now;
        confident_level = 1.0;
        updated_count += 1; 
        // std::cout<<"position_prev_ after: ("<< states_prev_(0,0)<<", "<<states_prev_(1,0)<< ")"<<std::endl;
        
        // std::cout<<"new prev_time_ - time_now:"<< prev_time_ - time_now <<std::endl;
        // std::cout<<"new states: ("<< states_now_(0,0)<<", "<<states_now_(1,0)<< " ,"<<states_now_(2,0)<<" ,"<< states_now_(3,0) <<" )"<<std::endl;


        // update history:
        x_history.push_back(states_now_(0,0));
        y_history.push_back(states_now_(1,0));

        last_measurement_x = measurement(0,0);
        last_measurement_y =measurement(1,0);
        last_measurement_width =  measurement(3,0);
        last_measurement_length =  measurement(4,0); 
        last_measurement_time = time_now;  

        // z_history.push_back(states_now_(2,0));
    }
    else{
        tracking_state = 3;
    }

}


void ObjectTracker::statePropagateOnly(double time_now){
    // using Contant Velocity model
    double delta_t = time_now - prev_time_;
    if ((delta_t < 10)){
        states_now_(0,0) = states_now_(0,0) + states_now_(2,0) * delta_t;
        states_now_(1,0) = states_now_(1,0) + states_now_(3,0) * delta_t;
        states_now_(2,0) = states_now_(2,0);
        states_now_(3,0) = states_now_(3,0);
        
        states_prev_ = states_now_;

        // update history:
        x_history.push_back(states_now_(0,0));
        y_history.push_back(states_now_(1,0));
        // z_history.push_back(states_now_(2,0));

        tracking_state = 3;
        prev_time_ = time_now;
    }
    confident_level = confident_level * (1.0 - 0.1);
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

void ObjectTracker::statePropagateOnlywithShape(double time_now){
    // using Contant Velocity model with shape
    double delta_t = time_now - prev_time_;
    if ((delta_t < 10)){
        states_now_(0,0) = states_now_(0,0) + states_now_(2,0) * delta_t;
        states_now_(1,0) = states_now_(1,0) + states_now_(3,0) * delta_t;
        // states_now_(2,0) = states_now_(2,0);
        // states_now_(3,0) = states_now_(3,0);
        // states_now_(4,0) = states_now_(4,0);
        // states_now_(5,0) = states_now_(5,0);

        states_prev_ = states_now_;

        // update history:
        x_history.push_back(states_now_(0,0));
        y_history.push_back(states_now_(1,0));
        // z_history.push_back(states_now_(2,0));

        tracking_state = 3;
        prev_time_ = time_now;
    }
    confident_level = confident_level * (1.0 - 0.1);
}

Eigen::MatrixXd ObjectTracker::statePropagationCVwithShape(Eigen::MatrixXd sigma_points,double delta_t){
    // Applied a Constant Velocity State Propagation

    Eigen::MatrixXd sigma_propagated(Eigen::MatrixXd(sigma_points.rows(),sigma_points.cols()));
    //Input should be (4*N) dimension
    for (int i = 0; i< sigma_points.cols();i++){
        sigma_propagated(0,i) = sigma_points(0,i) + sigma_points(2,i) * delta_t;//update x(t) = x(t-1) + delta_t * x_vel(t-1)
        sigma_propagated(1,i) = sigma_points(1,i) + sigma_points(3,i) * delta_t;//update y(t) = y(t-1) + delta_t * y_vel(t-1)
        sigma_propagated(2,i) = sigma_points(2,i); //update x_vel(t) = x_vel(t-1)
        sigma_propagated(3,i) = sigma_points(3,i); //update y_vel(t) = y_vel(t-1)
        sigma_propagated(4,i) = sigma_points(4,i); //update width(t) = width(t-1)
        sigma_propagated(5,i) = sigma_points(5,i); //update height(t) = height(t-1)
    }

    return sigma_propagated;
}

Eigen::MatrixXd ObjectTracker::measurementPropagationCVwithShape(Eigen::MatrixXd sigma_points,double delta_t){
    // Apply measurement update with x,y,width,height value

    Eigen::MatrixXd sigma_propagated(Eigen::MatrixXd(4,sigma_points.cols()));

    for (int i=0;i<sigma_points.cols();i++){
        sigma_propagated(0,i) = sigma_points(0,i); //x_measurement
        sigma_propagated(1,i) = sigma_points(1,i); //y_measurement
        sigma_propagated(2,i) = sigma_points(4,i); //width measurement
        sigma_propagated(3,i) = sigma_points(5,i); //length measurement
    }
    // std::cout<<"Sigma_propagated: "<<sigma_propagated<<std::endl;
    
    return sigma_propagated;
}

float clip(float n, float lower, float upper) {
    // helper function to clip a number
  return std::max(lower, std::min(n, upper));
}

Eigen::MatrixXd ObjectTracker::statePropagationCA(Eigen::MatrixXd sigma_points,double delta_t){
        // Applied a Constant Velocity State Propagation

    Eigen::MatrixXd sigma_propagated(Eigen::MatrixXd(sigma_points.rows(),sigma_points.cols()));
    //Input should be (4*N) dimension
    for (int i = 0; i< sigma_points.cols();i++){
        sigma_propagated(0,i) = sigma_points(0,i) + sigma_points(2,i) * delta_t + 0.5 * delta_t * delta_t * sigma_points(6,i);//update x(t) = x(t-1) + delta_t * x_vel(t-1) + 0.5 * delta_t^2 * acc_x(t-1)
        sigma_propagated(1,i) = sigma_points(1,i) + sigma_points(3,i) * delta_t + 0.5 * delta_t * delta_t * sigma_points(7,i);//update y(t) = y(t-1) + delta_t * y_vel(t-1) + 0.5 * delta_t^2 * acc_y(t-1)
        sigma_propagated(2,i) = sigma_points(2,i) + delta_t * sigma_points(6,i); //update x_vel(t) = x_vel(t-1) + delta_t * acc_x(t-1)
        sigma_propagated(3,i) = sigma_points(3,i) + delta_t * sigma_points(7,i); //update y_vel(t) = y_vel(t-1) + delta_t * acc_y(t-1)
        sigma_propagated(4,i) = sigma_points(4,i); //update width(t) = width(t-1)
        sigma_propagated(5,i) = sigma_points(5,i); //update height(t) = height(t-1)
        sigma_propagated(6,i) = sigma_points(6,i); //update acc_x(t) = acc_x(t-1)
        sigma_propagated(7,i) = sigma_points(7,i);  //update acc_y(t) = acc_y(t-1)
    }

    return sigma_propagated;
}


Eigen::MatrixXd ObjectTracker::measurementPropagationCA(Eigen::MatrixXd sigma_points,double delta_t){
    Eigen::MatrixXd sigma_propagated(Eigen::MatrixXd(4,sigma_points.cols()));

    for (int i=0;i<sigma_points.cols();i++){
        sigma_propagated(0,i) = sigma_points(0,i); //x_measurement
        sigma_propagated(1,i) = sigma_points(1,i); //y_measurement
        sigma_propagated(2,i) = sigma_points(4,i); //width measurement
        sigma_propagated(3,i) = sigma_points(5,i); //length measurement
    }
    // std::cout<<"Sigma_propagated: "<<sigma_propagated<<std::endl;
    
    return sigma_propagated;
}

void ObjectTracker::statePropagateOnlyCA(double time_now){
    double delta_t = time_now - prev_time_;
    if ((delta_t < 10)){
        states_now_(0,0) = states_now_(0,0) + states_now_(2,0) * delta_t + 0.5 * delta_t  * states_now_(6,0);
        states_now_(1,0) = states_now_(1,0) + states_now_(3,0) * delta_t + 0.5 * delta_t  * states_now_(7,0);
        states_now_(2,0) = states_now_(2,0) + delta_t  * states_now_(6,0);
        states_now_(3,0) = states_now_(3,0) + delta_t  * states_now_(7,0);
        
        states_prev_ = states_now_;

        // update history:
        x_history.push_back(states_now_(0,0));
        y_history.push_back(states_now_(1,0));
        // z_history.push_back(states_now_(2,0));

        tracking_state = 3;
        prev_time_ = time_now;
    }
    confident_level = confident_level * (1.0 - 0.1);
}