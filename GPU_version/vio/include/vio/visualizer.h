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

#ifndef VIO_VISUALIZER_H_
#define VIO_VISUALIZER_H_

#include <queue>
#include <ros/ros.h>
#include <vio/global.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <vio/ukf.h>

namespace vio {

class Frame;
class Point;
class Map;
class FrameHandlerMono;

typedef std::shared_ptr<Frame> FramePtr;

/// This class bundles all functions to publish visualisation messages.
class Visualizer
{
public:
  ros::NodeHandle pnh_;
  size_t trace_id_;
  ros::Publisher pub_frames_;
  ros::Publisher pub_points_;
  ros::Publisher pub_pose_with_cov_;

  Visualizer();

  ~Visualizer() {};

  void publishMinimal(
      UKF& ukf,
      const double timestamp);

};

} // end namespace vio

#endif /* VIO_VISUALIZER_H_ */
