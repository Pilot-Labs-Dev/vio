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

#ifndef VIO_POSE_OPTIMIZER_H_
#define VIO_POSE_OPTIMIZER_H_

#include <vio/global.h>
#include <vio/map.h>

namespace vio {

using namespace Eigen;
using namespace Sophus;
using namespace std;

typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<double,2,6> Matrix26d;
typedef Matrix<double,6,1> Vector6d;

class Point;

/// Motion-only bundle adjustment. Minimize the reprojection error of a single frame.
namespace pose_optimizer {

void optimizeGaussNewton(
    const size_t n_iter,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs,
    vio::Map& map,
    FILE* log);

} // namespace pose_optimizer
} // namespace vio

#endif // VIO_POSE_OPTIMIZER_H_
