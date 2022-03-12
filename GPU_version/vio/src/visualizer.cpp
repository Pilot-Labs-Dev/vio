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

#include <vio/visualizer.h>
#include <vio/frame_handler_mono.h>
#include <vio/point.h>
#include <vio/map.h>
#include <vio/feature.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <vio/timer.h>
#include <vio/output_helper.h>
#include <deque>
#include <vio/ukf.h>

namespace vio {

Visualizer::
Visualizer() :
    pnh_("~"),
    trace_id_(0)
{
  // Init ROS Marker Publishers
  //pub_frames_ = pnh_.advertise<visualization_msgs::Marker>("keyframes", 10);
  pub_points_ = pnh_.advertise<visualization_msgs::Marker>("points", 1000);
  pub_pose_with_cov_ = pnh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose_cov",10);
}

void Visualizer::publishMinimal(
        UKF& ukf,
    const double timestamp)
{
  ++trace_id_;
  std_msgs::Header header_msg;
  header_msg.frame_id = "/world";
  header_msg.seq = trace_id_;
  header_msg.stamp = ros::Time(timestamp);
  if(pub_pose_with_cov_.getNumSubscribers() > 0)
  {
    // publish cam in world frame (Estimated odometry in the worls frame)
    auto odom=ukf.get_location();
    //Rviz frame z up, x  right, y forward -> right hands (yaw counts from y)
    //Camera frame z front, x right, y down -> right hands (pitch counts from z)
    //IMU frame y front, x right, z up -> left hands (theta counts from y)
    Quaterniond q(AngleAxisd((odom.second.pitch()+M_PI_2), Vector3d::UnitZ()));
    geometry_msgs::PoseWithCovarianceStampedPtr msg_pose_with_cov(new geometry_msgs::PoseWithCovarianceStamped);
      msg_pose_with_cov->header = header_msg;
      msg_pose_with_cov->pose.pose.position.x = odom.second.se2().translation()(0);
      msg_pose_with_cov->pose.pose.position.y = odom.second.se2().translation()(1);
      msg_pose_with_cov->pose.pose.position.z = 0.0;
      msg_pose_with_cov->pose.pose.orientation.x = q.x();
      msg_pose_with_cov->pose.pose.orientation.y = q.y();
      msg_pose_with_cov->pose.pose.orientation.z = q.z();
      msg_pose_with_cov->pose.pose.orientation.w = q.w();
        msg_pose_with_cov->pose.covariance = {odom.first(0,0),odom.first(0,1),0.0,0.0,odom.first(0,2),0.0,
                                              odom.first(1,0),odom.first(1,1),0.0,0.0,odom.first(1,2),0.0,
                                              0.0,0.0,0.0000001,0.0,0.0,0.0,
                                              0.0,0.0,0.0,0.0000001,0.0,0.0,
                                              odom.first(2,0),odom.first(2,1),0.0,0.0,odom.first(2,2),0.0,
                                              0.0,0.0,0.0,0.0,0.0,0.0000001};
    pub_pose_with_cov_.publish(msg_pose_with_cov);
    if(pub_frames_.getNumSubscribers() > 0 || pub_points_.getNumSubscribers() > 0){
          publishPointMarker(
                  pub_points_, Eigen::Vector3d(odom.second.se2().translation()(0),odom.second.se2().translation()(1),0.0), "trajectory",
                  ros::Time::now(), trace_id_, 0, 0.01, Vector3d(0.,0.,0.5));
    }
  }

}
} // end namespace vio
