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

#include <vio/params_helper.h>
#include <vio/config.h>

namespace vio {

Config::Config() :
    trace_name(vk::getParam<string>("vio/trace_name", "VIO")),
    trace_dir(vk::getParam<string>("vio/trace_dir", "/tmp")),
    n_pyr_levels(vk::getParam<int>("vio/n_pyr_levels", 3)),
    use_imu(vk::getParam<bool>("vio/use_imu", false)),
    core_n_kfs(vk::getParam<int>("vio/core_n_kfs", 3)),
    map_scale(vk::getParam<double>("vio/map_scale", 1.0)),
    grid_size(vk::getParam<int>("vio/grid_size", 10)),
    init_min_disparity(vk::getParam<double>("vio/init_min_disparity", 50.0)),
    init_min_tracked(vk::getParam<int>("vio/init_min_tracked", 50)),
    init_min_inliers(vk::getParam<int>("vio/init_min_inliers", 40)),
    klt_max_level(vk::getParam<int>("vio/klt_max_level", 4)),
    klt_min_level(vk::getParam<int>("vio/klt_min_level", 2)),
    reproj_thresh(vk::getParam<double>("vio/reproj_thresh", 2.0)),
    poseoptim_thresh(vk::getParam<double>("vio/poseoptim_thresh", 2.0)),
    poseoptim_num_iter(vk::getParam<int>("vio/poseoptim_num_iter", 10)),
    structureoptim_max_pts(vk::getParam<int>("vio/structureoptim_max_pts", 20)),
    structureoptim_num_iter(vk::getParam<int>("vio/structureoptim_num_iter", 10)),
    loba_thresh(vk::getParam<double>("vio/loba_thresh", 2.0)),
    loba_robust_huber_width(vk::getParam<double>("vio/loba_robust_huber_width", 1.0)),
    loba_num_iter(vk::getParam<int>("vio/loba_num_iter", 0)),
    kfselect_mindist(vk::getParam<double>("vio/kfselect_mindist", 0.12)),
    triang_min_corner_score(vk::getParam<double>("vio/triang_min_corner_score", 20.0)),
    triang_half_patch_size(vk::getParam<int>("vio/triang_half_patch_size", 4)),
    subpix_n_iter(vk::getParam<int>("vio/subpix_n_iter", 10)),
    max_n_kfs(vk::getParam<int>("vio/max_n_kfs", 10)),
    img_imu_delay(vk::getParam<double>("vio/img_imu_delay", 0.0)),
    max_fts(vk::getParam<int>("vio/max_fts", 120)),
    quality_min_fts(vk::getParam<int>("vio/quality_min_fts", 50)),
    quality_max_drop_fts(vk::getParam<int>("vio/quality_max_drop_fts", 40)),
    ACC_noise(vk::getParam<double>("vio/ACC_ekf", 1e-4)),
    GYO_noise(vk::getParam<double>("vio/GYO_ekf", 1e-4)),
    vo_ekf_t(vk::getParam<double>("vio/VO_ekf_translation", 1e-4)),
    vo_ekf_o(vk::getParam<double>("vio/VO_ekf_orientation", 1e-4)),
    cmd_ekf_t(vk::getParam<double>("vio/CMD_ekf_translation", 1e-4)),
    cmd_ekf_o(vk::getParam<double>("vio/CMD_ekf_orientation", 1e-4))
{}

Config& Config::getInstance()
{
  static Config instance; // Instantiated on first use and guaranteed to be destroyed
  return instance;
}

} // namespace vio

