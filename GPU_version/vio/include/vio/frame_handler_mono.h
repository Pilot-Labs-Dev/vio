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

#ifndef VIO_FRAME_HANDLER_H_
#define VIO_FRAME_HANDLER_H_

#include <set>
#include <vio/abstract_camera.h>
#include <vio/frame_handler_base.h>
#include <vio/reprojector.h>
#include <vio/initialization.h>
#include <vio/ukf.h>
#include <vio/cl_class.h>
#include <vio/global_optimizer.h>

namespace vio {

/// Monocular Visual Odometry Pipeline as described in the VIO paper.
class FrameHandlerMono : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FrameHandlerMono(vk::AbstractCamera* cam,Eigen::Matrix<double,3,1>& init);
  virtual ~FrameHandlerMono();

  /// Provide an image.
  void addImage(const cv::Mat& img, double timestamp,const ros::Time& time);


  /// Access the depth filter.
  BA_Glob* depthFilter() const{ return ba_glob_; }


  void UpdateIMU(double* value,const ros::Time& time);
  void UpdateCmd(double* value,const ros::Time& time);
  UKF ukfPtr_;
#if VIO_DEBUG
        FILE* log_=nullptr;
#endif
protected:
  vk::AbstractCamera* cam_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
  Reprojector reprojector_;                     //!< Projects points from other keyframes into the current frame
  FramePtr new_frame_;                          //!< Current frame.
  FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
  vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?
  initialization::KltHomographyInit* klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.
  BA_Glob* ba_glob_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.
  opencl* gpu_fast_;
  ros::Time time_;



  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first frame until a keyframe is selected.
  virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();


  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf();

};

} // namespace vio

#endif // VIO_FRAME_HANDLER_H_
