// 0.1 m/s linear velocity
// 0.2 radian/s angular velocity
#include <ros/ros.h>
#include <string>
#include <chrono>
#include <boost/thread.hpp>
#include <vio/frame_handler_mono.h>
#include <vio/params_helper.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Twist.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <vio/abstract_camera.h>
#include <vio/camera_loader.h>
#include <vio/frame.h>
#include <vio/getOdom.h>
#include <vio/start.h>
#include <vio/stop.h>
#include <vio/global_optimizer.h>
#include <ros/callback_queue.h>
#if VIO_DEBUG
#include <vio/visualizer.h>
#endif

class VioNode
{
public:
      vio::FrameHandlerMono* vo_;
      vk::AbstractCamera* cam_;
      VioNode();
      ~VioNode();
      void imgCb(const sensor_msgs::ImageConstPtr& msg);
      void imuCb(const sensor_msgs::ImuPtr& imu);
    void cmdCb(const geometry_msgs::TwistPtr &imu);
    void imu_th();
    bool getOdom(vio::getOdom::Request& req, vio::getOdom::Response& res);
    uint trace_id_= 0;
    bool start_=false;
    bool start(vio::start::Request& req, vio::start::Response& res){
        if(req.on==1 && !start_){
            start_=true;
            if(vo_->stage()!=vio::FrameHandlerBase::STAGE_DEFAULT_FRAME) {
                vo_->depthFilter()->startThread();
                vo_->start();
            }else{
                vo_->depthFilter()->startThread();
                vo_->reset();
            }
            imu_the_=new boost::thread(&VioNode::imu_th,this);
            ++trace_id_;
            res.ret=0;
        }else{
            res.ret=100;
        }
        return true;
    };

    bool stop(vio::stop::Request& req, vio::stop::Response& res){
        if(req.off==1 && imu_the_!=NULL){
            start_=false;
            vo_->depthFilter()->stopThread();
#if VIO_DEBUG
    fclose(vo_->log_);
#endif
            imu_the_->interrupt();
            if(imu_the_->get_id()==boost::this_thread::get_id())
                imu_the_->detach();
            else
                imu_the_->join();
            imu_the_=NULL;
            res.ret=0;
        }else{
            res.ret=100;
        }
        return true;
    };
private:
    double* imu_;
    size_t cam_syn_=2;
    ros::Time imu_time_;
    int width;
    int height;
    boost::thread* imu_the_= nullptr;
#if VIO_DEBUG
    vio::Visualizer visualizer_;
#endif
};

VioNode::VioNode() :
      vo_(NULL),
      cam_(NULL)
{
    width=vk::getParam<int>("vio/cam_width", 640);
    height=vk::getParam<int>("vio/cam_height", 480);
    imu_=(double *)calloc(static_cast<std::size_t>(3), sizeof(double));
    if(!vk::camera_loader::loadFromRosNs("vio", cam_))
        throw std::runtime_error("Camera model not correctly specified.");
    Eigen::Matrix<double,3,1> init;
    init<<1e-19,1e-19,1e-19;
    vo_ = new vio::FrameHandlerMono(cam_,init);
    usleep(500);
}

VioNode::~VioNode()
{
    vo_->depthFilter()->stopThread();
    start_=false;
    usleep(5000);
    imu_the_->join();
    delete imu_the_;
    delete vo_;
    delete cam_;
}

void VioNode::imgCb(const sensor_msgs::ImageConstPtr& msg)
{

    if(!start_)return;
      try {
          cv::Mat img=cv_bridge::toCvShare(msg, "mono8")->image;
          if(img.empty())return;
          cv::Mat imgbul,float_img,frame,show;//tem;
          img.convertTo(float_img,CV_64F,1.f/255);
          float_img*=2.0;
          float_img+=0.2;
          float_img.convertTo(frame,CV_8UC1,255);
          cv::Laplacian(frame,imgbul,CV_64F);
          cv::Scalar mean, stddev;
          meanStdDev(imgbul, mean, stddev, cv::Mat());
/*          cv::cvtColor(frame,show,CV_GRAY2RGB);
          imshow("New Image", show);
          cv::waitKey(10);*/
          if(stddev.val[0] * stddev.val[0]< 30.0){
              ROS_WARN("Frame is blur or too dark");
              return;
          }
          vo_->addImage(frame, msg->header.stamp.toSec(),msg->header.stamp);
      } catch (cv_bridge::Exception& e) {
        ROS_ERROR("vo exception: %s", e.what());
      }
}
void VioNode::imuCb(const sensor_msgs::ImuPtr &imu) {
    if(!start_)return;
    double imu_in[3];
    imu_in[0] = 0.2*imu_[0]+0.8*imu->linear_acceleration.x;
    imu_in[1] = 0.2*imu_[1]+0.8*imu->linear_acceleration.y;
    imu_in[2] = 0.2*imu_[2]+0.8*imu->angular_velocity.z;
    memcpy(imu_, imu_in, static_cast<std::size_t>(3*sizeof(double)));
    vo_->UpdateIMU(imu_in,imu->header.stamp);
    imu_time_=imu->header.stamp;
#if VIO_DEBUG
    visualizer_.publishMinimal(vo_->ukfPtr_, imu->header.stamp.toSec());
#endif
}
void VioNode::cmdCb(const geometry_msgs::TwistPtr &cmd) {
    if(!start_)return;
    double _cmd[3]={cmd->linear.x,cmd->linear.y,cmd->angular.z};
    vo_->UpdateCmd(_cmd,imu_time_);
#if VIO_DEBUG
    auto odom=vo_->ukfPtr_.get_location();
    fprintf(vo_->log_,"[%s] Odometry x=%f, y=%f, theta=%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            odom.second.se2().translation()(0),odom.second.se2().translation()(1),
            odom.second.pitch());
#endif
}
void VioNode::imu_th(){
    ros::NodeHandle nh;
    ros::CallbackQueue Q;
    nh.setCallbackQueue(&Q);
    ros::Subscriber imu_sub=nh.subscribe(vk::getParam<std::string>("vio/imu_topic", "imu/raw"),1,&VioNode::imuCb, this,ros::TransportHints().tcpNoDelay());
    ros::Subscriber cmd_sub=nh.subscribe(vk::getParam<std::string>("vio/cmd_topic", "cmd/raw"),1,&VioNode::cmdCb, this,ros::TransportHints().tcpNoDelay());
    while(start_ && !boost::this_thread::interruption_requested())
    {
        Q.callOne(ros::WallDuration(10,0.0));
        usleep(5000);
    }

}
bool VioNode::getOdom(vio::getOdom::Request &req, vio::getOdom::Response &res) {
    if(req.get==1){
        res.header.stamp = ros::Time();
        res.header.frame_id = "/world";
        res.header.seq = trace_id_;
        auto odom=vo_->ukfPtr_.get_location();
        //Camera frame z front, x right, y down -> right hands (pitch counts from z)
        //IMU frame y front, x right, z up -> left hands (theta counts from y)
        res.x=odom.second.se2().translation()(0);
        res.y=odom.second.se2().translation()(1);
        res.yaw=-odom.second.pitch();
        res.cov={odom.first(0,0),odom.first(0,1),odom.first(0,2),
                 odom.first(1,0),odom.first(1,1),odom.first(1,2),
                 odom.first(2,0),odom.first(2,1),odom.first(2,2)};
    }
    return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vio");
  ros::NodeHandle nh;
  ros::CallbackQueue Q;
  nh.setCallbackQueue(&Q);
  VioNode vio;
  ros::ServiceServer start = nh.advertiseService(vk::getParam<std::string>("vio/start", "start"),
                                                 &VioNode::start, &vio);
  ros::ServiceServer stop = nh.advertiseService(vk::getParam<std::string>("vio/stop", "stop"),
                                                &VioNode::stop, &vio);
  ros::ServiceServer getOdom = nh.advertiseService(vk::getParam<std::string>("vio/odom", "get_odom"),
                                                   &VioNode::getOdom, &vio);
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber it_sub = it.subscribe(vk::getParam<std::string>("vio/cam_topic", "camera/image_raw"), 1, &VioNode::imgCb, &vio);
  // start processing callbacks
  while(ros::ok())
  {
      Q.callOne(ros::WallDuration(30,0.0));
  }
  printf("VIO terminated.\n");
  return 0;
}
