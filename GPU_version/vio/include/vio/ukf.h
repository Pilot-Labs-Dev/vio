//
// Created by root on 6/8/21.
//
#ifndef VIO_UKF_H
#define VIO_UKF_H

#include <iostream>
#include <fstream>
#include <vio/config.h>
#include <Eigen/Dense>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <chrono>
#include <ctime>
class Base{
public:
    Base(const Eigen::Matrix<double, 3, 1> &init) {
        state_=Eigen::Vector4d(init.x(),init.y(),init.z(),0.0);
        cov_.setIdentity();
        cov_*=0.0001;
        state_h_=Eigen::Vector4d(init.x(),init.y(),init.z(),0.0);
        cov_h_.setIdentity();
        cov_h_*=0.0001;
    };
    virtual ~Base(){
    };
    void predict(double dx,double dy,double dpitch,double time){
        double dt=0.2;
        state_h_(2)=state_(2)+dt*dpitch;
        state_h_(1)=state_(1)+dt*(dy*cos(state_(2))-dx*sin(state_(2)));
        state_h_(0)=state_(0)+dt*(dx*cos(state_(2))+dy*sin(state_(2)));
        state_h_(3)=dpitch;
        Eigen::Matrix<double,4,4> R;
        R<<vio::Config::Cmd_ekf_t(),0,0,0,
           0,vio::Config::Cmd_ekf_t(),0,0,
           0,0,vio::Config::Cmd_ekf_o(),0,
           0,0,0,vio::Config::Cmd_ekf_o();
        Eigen::Matrix<double,4,4> G;
        G<<1.0,0,dt*(-dx*sin(state_(2))+dy*cos(state_(2))),0.0,
           0,1.0,dt*(-dy*sin(state_(2))-dx*cos(state_(2))),0.0,
           0,0,1.0,dt,
           0,0,0,1.0;
        cov_h_=G*cov_*G.transpose()+R;
        setTime();
    }
    void correct(double ddx, double ddy, double dpitch, double time){
        auto imu_end = std::chrono::steady_clock::now();
        double dt=1e-9*(imu_end-vio_t).count();
        vio_t=std::chrono::steady_clock::now();
        Eigen::Matrix<double,4,4> H;
        Eigen::Matrix<double,4,4> Q;
        Eigen::Matrix<double,4,1> E;
        Eigen::Matrix<double,4,4> k;
        //update
        E(0)=state_(0)+pow(dt,2)*(ddx*cos(state_(2))+ddy*sin(state_(2)))-state_h_(0);
        E(1)=state_(1)+pow(dt,2)*(ddy*cos(state_(2))-ddx*sin(state_(2)))-state_h_(1);
        E(2)=state_(2)+dt*state_(3)-state_h_(2);
        E(3)=dpitch-state_h_(3);
        H<<1.0,0,pow(dt,2)*(ddy*cos(state_(2))-ddx*sin(state_(2))),0.0,
           0,1.0,-pow(dt,2)*(ddx*cos(state_(2))+ddy*sin(state_(2))),0.0,
           0,0,1.0,dt,
           0,0,0,1.0;
        Q<<vio::Config::ACC_Noise(),0,0,0,
                0,vio::Config::ACC_Noise(),0,0,
                0,0,vio::Config::GYO_Noise(),0,
                0,0,0,vio::Config::GYO_Noise();
        k=cov_h_*H.transpose()*(H*cov_h_*H.transpose()+Q).inverse();
        state_=state_h_+k*E;
        cov_=(Eigen::MatrixXd::Identity(4,4)-k*H)*cov_h_;
        //predict
        state_h_(2)=state_(2)+dt*state_(3);
        state_h_(1)=state_(1)+pow(dt,2)*(ddy*cos(state_(2))-ddx*sin(state_(2)));
        state_h_(0)=state_(0)+pow(dt,2)*(ddx*cos(state_(2))+ddy*sin(state_(2)));
        state_h_(3)=state_(3);
        Eigen::Matrix<double,4,4> R;
        R<<vio::Config::ACC_Noise(),0,0,0,
                0,vio::Config::ACC_Noise(),0,0,
                0,0,vio::Config::GYO_Noise(),0,
                0,0,0,vio::Config::GYO_Noise();
        Eigen::Matrix<double,4,4> G;
        G<<1.0,0,pow(dt,2)*(ddy*cos(state_(2))-ddx*sin(state_(2))),0,
           0,1.0,pow(dt,2)*(ddx*cos(state_(2))+ddy*sin(state_(2))),0,
           0,0,1.0,dt,
           0,0,0,1.0;
        cov_h_=G*cov_*G.transpose()+R;

    }
    void correct(double x, double z,double pitch){
        Eigen::Matrix<double,3,4> H;
        Eigen::Matrix<double,3,3> Q;
        Eigen::Matrix<double,3,1> E;
        Eigen::Matrix<double,4,3> k;
        H<<1.0,0,0,0,
           0,1.0,0,0,
           0,0,1.0,0;
        Q<<vio::Config::VO_ekf_t(),0,0,
           0,vio::Config::VO_ekf_t(),0,
           0,0,vio::Config::VO_ekf_o();
        E<<x-state_h_(0),z-state_h_(1),pitch-state_h_(2);
        k=cov_h_*H.transpose()*(H*cov_h_*H.transpose()+Q).inverse();
        state_=state_h_+(k*E);
        cov_=(Eigen::MatrixXd::Identity(4,4)-k*H)*cov_h_;
    }
    void setTime(){
        vio_t=std::chrono::steady_clock::now();
    }
    Eigen::Matrix<double,4,4> cov_;
    Eigen::Matrix<double, 4, 1> state_;
private:
    std::chrono::time_point<std::chrono::steady_clock,std::chrono::duration<double>> vio_t;
    Eigen::Matrix<double,4,4> cov_h_;
    Eigen::Matrix<double, 4, 1> state_h_;//state{x,y,pitch( rotation around z),dyaw} in world frame

};

class UKF {
public:
    UKF(const Eigen::Matrix<double, 3, 1>& init) : filter_(new Base(init)) {
    };
    virtual ~UKF() {
        delete filter_;
    };
    void UpdateIMU(double x/*in imu frame*/,double y/*in imu frame*/,double theta/*in imu frame*/,const ros::Time& time) {
       /* if(abs(x)<1.0)x=pow(x,3);//picked up from your code
        if(abs(y)<1.0)y=pow(y,3);//picked up from your code*/
        ++imu_syn_count_;
        if(imu_syn_[imu_syn_count_ % 10]){
            imu_syn_[imu_syn_count_ % 10]=false;
        }else{
            return;
        }
        if(imu_syn_count_>50)imu_syn_count_=0;
        boost::unique_lock<boost::mutex> lock(ekf_mut_);
        filter_->correct(x,y,theta,1e-9*time.toNSec());
    };
    //IMU frame y front, x right, z up -> left hands (theta counts from y)
    void UpdateCmd(double x/*in imu frame*/,double y/*in imu frame*/,double theta/*in imu frame*/,const ros::Time& time) {
        imu_syn_[cmd_syn_count_ % 10]=true;
        ++cmd_syn_count_;
        if(cmd_syn_count_>50)cmd_syn_count_=0;
        boost::unique_lock<boost::mutex> lock(ekf_mut_);
        filter_->predict(x,y,theta,1e-9*time.toNSec());
    };
    //Camera frame z back, x right, y down -> left hands (pitch counts from x)correct
    //Notice T_F_W is the position of the first frame with respect to the new frame while they are looking at one feature
    std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5> UpdateVO(double x/*in camera frame*/,double z/*in camera frame*/,double pitch/*in camera frame*/) {
        boost::unique_lock<boost::mutex> lock(ekf_mut_);
        filter_->correct(x,z,-1.0*pitch);
        Eigen::Matrix<double,3,3> cov;
        cov<<filter_->cov_(0,0),filter_->cov_(0,1),filter_->cov_(0,2),
                filter_->cov_(1,0),filter_->cov_(1,1),filter_->cov_(1,2),
                filter_->cov_(2,0),filter_->cov_(2,1),filter_->cov_(2,2);
        return std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5>(cov,vio::SE2_5(filter_->state_(0),filter_->state_(1),-1.0*(filter_->state_(2))));
    };
    std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5> get_location(){
        boost::unique_lock<boost::mutex> lock(ekf_mut_);
        Eigen::Matrix<double,3,3> cov;
        cov<<filter_->cov_(0,0),filter_->cov_(0,1),filter_->cov_(0,2),
                filter_->cov_(1,0),filter_->cov_(1,1),filter_->cov_(1,2),
                filter_->cov_(2,0),filter_->cov_(2,1),filter_->cov_(2,2);
        return std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5>(cov,vio::SE2_5(filter_->state_(0),filter_->state_(1),-1.0*(filter_->state_(2))));
    }
    void setImuTime(){
        filter_->setTime();
    }
    size_t cmd_syn_count_=0;
    size_t imu_syn_count_=2;
    bool imu_syn_[10]={false,false,false,false,false,false,false,false,false,false};
private:
     Base* filter_= nullptr;
    boost::mutex ekf_mut_;
};
#endif //VIO_UKF_H
