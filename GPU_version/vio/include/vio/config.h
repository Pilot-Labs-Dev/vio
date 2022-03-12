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

#ifndef VIO_CONFIG_H_
#define VIO_CONFIG_H_

#include <string>
#include <stdint.h>
#include <stdio.h>

namespace vio {

using std::string;

/// Global configuration file of VIO.
/// Implements the Singleton design pattern to allow global access and to ensure
/// that only one instance exists.
class Config
{
public:
  static Config& getInstance();


  /// Number of pyramid levels used for features.
  static size_t& nPyrLevels() { return getInstance().n_pyr_levels; }


  /// Feature grid size of a cell in [px].
  static size_t& gridSize() { return getInstance().grid_size; }

  /// Initialization: Minimum required disparity between the first two frames.
  static double& initMinDisparity() { return getInstance().init_min_disparity; }


  /// Initialization: Minimum number of inliers after RANSAC.
  static size_t& initMinInliers() { return getInstance().init_min_inliers; }

  /// Maximum level of the Lucas Kanade tracker.
  static size_t& kltMaxLevel() { return getInstance().klt_max_level; }

  /// Minimum level of the Lucas Kanade tracker.
  static size_t& kltMinLevel() { return getInstance().klt_min_level; }


  /// Reprojection threshold after pose optimization.
  static double& poseOptimThresh() { return getInstance().poseoptim_thresh; }


  /// Maximum number of points to optimize at every iteration.
  static size_t& structureOptimMaxPts() { return getInstance().structureoptim_max_pts; }

  /// Number of iterations in structure optimization.
  static size_t& structureOptimNumIter() { return getInstance().structureoptim_num_iter; }

  /// Reprojection threshold after bundle adjustment.

  /// Threshold for the robust Huber kernel of the local bundle adjustment.
  static double& lobaRobustHuberWidth() { return getInstance().loba_robust_huber_width; }

  /// Number of iterations in the local bundle adjustment.
  static size_t& lobaNumIter() { return getInstance().loba_num_iter; }

  /// Select only features with a minimum Harris corner score for triangulation.
  static double& triangMinCornerScore() { return getInstance().triang_min_corner_score; }


  /// Limit the number of keyframes in the map. This makes nslam essentially.
  /// a Visual Odometry. Set to 0 if unlimited number of keyframes are allowed.
  /// Minimum number of keyframes is 3.
  static size_t& maxNKfs() { return getInstance().max_n_kfs; }


  /// acc white noise in continuous.
  static double& ACC_Noise() { return getInstance().ACC_noise; }

  /// Gro white noise in continuous.
  static double& GYO_Noise() { return getInstance().GYO_noise; }
  static double& VO_ekf_t() { return getInstance().vo_ekf_t; }
  static double& VO_ekf_o() { return getInstance().vo_ekf_o; }
  static double& Cmd_ekf_t() { return getInstance().cmd_ekf_t; }
  static double& Cmd_ekf_o() { return getInstance().cmd_ekf_o; }


private:
  Config();
  Config(Config const&);
  void operator=(Config const&);
  string trace_name;
  string trace_dir;
  size_t n_pyr_levels;
  bool use_imu;
  size_t core_n_kfs;
  double map_scale;
  size_t grid_size;
  double init_min_disparity;
  size_t init_min_tracked;
  size_t init_min_inliers;
  size_t klt_max_level;
  size_t klt_min_level;
  double reproj_thresh;
  double poseoptim_thresh;
  size_t poseoptim_num_iter;
  size_t structureoptim_max_pts;
  size_t structureoptim_num_iter;
  double loba_thresh;
  double loba_robust_huber_width;
  size_t loba_num_iter;
  double kfselect_mindist;
  double triang_min_corner_score;
  size_t triang_half_patch_size;
  size_t subpix_n_iter;
  size_t max_n_kfs;
  double img_imu_delay;
  size_t max_fts;
  size_t quality_min_fts;
  int quality_max_drop_fts;
  double ACC_noise,GYO_noise,vo_ekf_t,vo_ekf_o,cmd_ekf_t,cmd_ekf_o;
};

} // namespace vio

#endif // VIO_CONFIG_H_
