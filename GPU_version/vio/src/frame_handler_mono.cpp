// This file is part of VIO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// VIO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// VIO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <vio/config.h>
#include <vio/frame_handler_mono.h>
#include <vio/map.h>
#include <vio/frame.h>
#include <vio/feature.h>
#include <vio/point.h>
#include <vio/pose_optimizer.h>
#include <vio/global_optimizer.h>
#include <vio/for_it.hpp>
#include <assert.h>
#if VIO_DEBUG
#include <sys/stat.h>
#endif

namespace vio {
FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam,Eigen::Matrix<double,3,1>& init) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  ba_glob_(NULL),
  ukfPtr_(init),
  time_(ros::Time::now())
{
    gpu_fast_= new opencl(cam_);
    gpu_fast_->make_kernel("fast_gray");
    gpu_fast_->make_kernel("compute_residual");
    initialize();
#if VIO_DEBUG
    log_ =fopen((std::string(PROJECT_DIR)+"/frame_handler_log.txt").c_str(),"w+");
    assert(log_);
    chmod((std::string(PROJECT_DIR)+"/frame_handler_log.txt").c_str(), ACCESSPERMS);
    klt_homography_init_=new initialization::KltHomographyInit(gpu_fast_,&ukfPtr_,log_);
#else
    klt_homography_init_=new initialization::KltHomographyInit(gpu_fast_&ukfPtr_);
#endif
}

void FrameHandlerMono::initialize()
{
  ba_glob_ = new BA_Glob(map_);
  ba_glob_->startThread();
}

FrameHandlerMono::~FrameHandlerMono()
{
  delete ba_glob_;
}

void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp,const ros::Time& time)
{
    ukfPtr_.setImuTime();
  if(!startFrameProcessingCommon(timestamp)){
      return;
  }
  // some cleanup from last iteration, can't do before because of visualization
  overlap_kfs_.clear();
  // create new frame
  new_frame_=std::make_shared<Frame>(cam_, img.clone(), timestamp);
  time_=time;
  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    res = processFirstFrame();
  last_frame_ = new_frame_;
  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
#if VIO_DEBUG
    fprintf(log_,"[%s] frame process finished the id is: %d the obs is:%d \n",vio::time_in_HH_MM_SS_MMM().c_str(),
            last_frame_->id_,last_frame_->nObs());
#endif

}

FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  if(klt_homography_init_->addFirstFrame(new_frame_) == initialization::FAILURE){
      return RESULT_NO_KEYFRAME;
  }
  //new_frame_->setKeyframe();
  //map_.reset();
  //map_.addKeyframe(new_frame_);
  stage_ = STAGE_SECOND_FRAME;
#if VIO_DEBUG
    fprintf(log_,"[%s] Init: Selected first frame. \n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
  initialization::InitResult res = klt_homography_init_->addSecondFrame(new_frame_);
#if VIO_DEBUG
    fprintf(log_,"[%s] Init: distance between the first and current frame is x:%f ,z=%f,angle between two frames: %f \n",vio::time_in_HH_MM_SS_MMM().c_str(),
            new_frame_->T_f_w_.se2().translation().x()-last_frame_->T_f_w_.se2().translation().x(),
            new_frame_->T_f_w_.se2().translation().y()-last_frame_->T_f_w_.se2().translation().y(),
            new_frame_->T_f_w_.pitch()-last_frame_->T_f_w_.pitch());
#endif
  if(res == initialization::NO_KEYFRAME){
      return RESULT_NO_KEYFRAME;
  }else if(res == initialization::FAILURE){
      stage_ = STAGE_FIRST_FRAME;
      return RESULT_FAILURE;
  }
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  klt_homography_init_->reset();
  double depth_mean, depth_min;
  new_frame_->getSceneDepth(map_,depth_mean, depth_min);
  // add frame to map
#if VIO_DEBUG
    fprintf(log_,"[%s] Init: Selected Second frame. \t The number of features: %d depth mean:%f min:%f\n",
                       vio::time_in_HH_MM_SS_MMM().c_str(),new_frame_->fts_.size(),depth_mean,depth_min);
#endif
  //ba_glob_->new_key_frame();
  ROS_INFO("VIO initialized :)");
  ROS_INFO("Running ...");
  return RESULT_IS_KEYFRAME;
}


FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
  auto init_f= ukfPtr_.get_location();
  new_frame_->T_f_w_=init_f.second;
  new_frame_->Cov_ = init_f.first;
  reprojector_.reprojectMap(new_frame_, last_frame_,overlap_kfs_, gpu_fast_, log_);
  int n_point=0;
  for(auto i:overlap_kfs_)n_point+=i.second;
#if VIO_DEBUG
    fprintf(log_,"[%s] After Reprojection Map nMatches:%d ,distance between ekf and vo x:%f ,z=%f,angle between two frames:%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            n_point,
            new_frame_->T_f_w_.se2().translation().x()-init_f.second.se2().translation().x(),
            new_frame_->T_f_w_.se2().translation().y()-init_f.second.se2().translation().y(),
            fabs(new_frame_->T_f_w_.pitch()-init_f.second.pitch()));
#endif
  boost::unique_lock< boost::mutex > lock(ba_glob_->ba_mux_);
  size_t sfba_n_edges_final=0;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
            10,
            new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final,map_,log_);
#if VIO_DEBUG
    fprintf(log_,"[%s] After pose optimization, distance between ekf and vo x:%f ,z=%f,angle between two frames:%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            new_frame_->T_f_w_.se2().translation().x()-init_f.second.se2().translation().x(),
            new_frame_->T_f_w_.se2().translation().y()-init_f.second.se2().translation().y(),
            fabs(new_frame_->T_f_w_.pitch()-init_f.second.pitch()));
#endif
    if((init_f.second.se2().translation()-new_frame_->T_f_w_.se2().translation()).norm()>0.2 ||
       fabs(new_frame_->T_f_w_.pitch()-init_f.second.pitch())>0.2 || sfba_n_edges_final<10){
        new_frame_=last_frame_;
        return RESULT_FAILURE;
    }
  auto result=ukfPtr_.UpdateVO(new_frame_->T_f_w_.se2().translation()(0),
                                new_frame_->T_f_w_.se2().translation()(1),new_frame_->T_f_w_.pitch());
  new_frame_->T_f_w_ =result.second;
  new_frame_->Cov_ = result.first;
#if VIO_DEBUG
    fprintf(log_,"[%s] Update EKF and 3D points the number of feature in the new frame: %d and number of obs: %d\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            new_frame_->fts_.size(),new_frame_->nObs());
#endif
  double depth_mean=0.0, depth_min=0.0;
    new_frame_->getSceneDepth(map_, depth_mean, depth_min);
#if VIO_DEBUG
    fprintf(log_,"[%s] frame Scene Depth mean:%f ,depth min:%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            depth_mean,
            depth_min);
#endif
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());

  // select keyframe

  if(!needNewKf())//edited
  {
        return RESULT_NO_KEYFRAME;
  }

  new_frame_->setKeyframe();
#if VIO_DEBUG
    fprintf(log_,"[%s] Choose frame as a key frame Scene Depth mean:%f ,depth min:%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            depth_mean,
            depth_min);
#endif
  // new keyframe selected
  for(auto&& it:new_frame_->fts_){
      if(it->point != NULL && !it->point->pos_.hasNaN() && it->point->pos_.norm() !=0.){
          it->point->addFrameRef(it);
      }else{
          map_.safeDeletePoint(it->point);
      }
  }

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    map_.safeDeleteFrame(furthest_frame);
  }
  // add keyframe to map
  map_.addKeyframe(new_frame_);
  if(map_.checkKeyFrames()){
      ba_glob_->new_key_frame();
/*      std::unique_ptr<feature_detection::FastDetector> detector=std::make_unique<feature_detection::FastDetector>(
              new_frame_->img().cols, new_frame_->img().rows, Config::gridSize(), gpu_fast_,Config::nPyrLevels());
      detector->detect(new_frame_, new_frame_->img_pyr_, Config::triangMinCornerScore(), new_frame_->fts_);*/
      return RESULT_IS_KEYFRAME;
  }
  return RESULT_FAILURE;
}


void FrameHandlerMono::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  overlap_kfs_.clear();
}
bool FrameHandlerMono::needNewKf()
{
    size_t com_obs=0;
    SE2_5 closest_kfs(0,0,0);
  for(auto&& it:overlap_kfs_)
  {
      if(it.first->id_==last_frame_->id_)continue;
      if(it.second>com_obs){
          com_obs=it.second;
          closest_kfs=SE2_5(it.first->T_f_w_.se2());
      }
  }
#if VIO_DEBUG
    fprintf(log_,"[%s] need key frame pitch dis: %f translation dif:%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            fabs(closest_kfs.pitch()-new_frame_->T_f_w_.pitch()),(closest_kfs.se2().translation()-new_frame_->T_f_w_.se2().translation()).norm());
#endif
  if(fabs(closest_kfs.pitch()-new_frame_->T_f_w_.pitch()) > 0.1 || fabs((closest_kfs.se2().translation()-new_frame_->T_f_w_.se2().translation()).norm())>0.1)return true;
  return false;
}
void FrameHandlerMono::UpdateIMU(double* value,const ros::Time& time){
    if(value== nullptr)return;
    ukfPtr_.UpdateIMU(value[0],value[1],value[2],time);
}
void FrameHandlerMono::UpdateCmd(double* value,const ros::Time& time){
    if(value== nullptr)return;
    ukfPtr_.UpdateCmd(value[0],value[1],value[2],time);
}

} // namespace vio
