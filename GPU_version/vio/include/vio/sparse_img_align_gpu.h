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

#ifndef VIO_SPARSE_IMG_ALIGN_GPU_H_
#define VIO_SPARSE_IMG_ALIGN_GPU_H_

#include <vio/nlls_solver.h>
#include <vio/global.h>
#include <vio/cl_class.h>
#include <vio/frame.h>
#include <vio/feature.h>
#include <vio/config.h>
#include <vio/point.h>
#include <vio/abstract_camera.h>
#include <vio/vision.h>
#include <vio/math_utils.h>
#include <algorithm>


namespace vio {

class Feature;

/// Optimize the pose of the frame by minimizing the photometric error of feature patches.
class SparseImgAlignGpu : public vk::NLLSSolver<3, SE2>
{
  static const int patch_halfsize_ = 12;
  static const int patch_size_ = 12*patch_halfsize_;
  static const int patch_area_ = patch_size_*patch_size_;
public:

  SparseImgAlignGpu(
      int n_levels,
      int min_level,
      int n_iter,
      Method method,
      bool verbose,
      opencl* residual);

  size_t run(
      FramePtr ref_frame,
      FramePtr cur_frame,
      FILE* log);

   // FILE* data= nullptr;

protected:
  int level_;                     //!< current pyramid level on which the optimization runs.
  int max_level_;                 //!< coarsest pyramid level for the alignment.
  int min_level_;                 //!< finest pyramid level for the alignment.
  size_t feature_counter_=0;
  opencl* residual_= nullptr;
  std::vector<bool> errors;

  // cache:
  std::vector<bool> visible_fts_;
  virtual double computeResiduals(const ModelType& model,
                           bool linearize_system,
                           bool compute_weight_scale) {
        return 0.0;
    };
  virtual double computeResiduals(bool linearize_system, bool compute_weight_scale = false);
  virtual bool solve();
  virtual void update();
  virtual void update(const ModelType& old_model, ModelType& new_model) { };
  virtual void startIteration();
  virtual void finishIteration();
};

} // namespace vio

#endif // VIO_SPARSE_IMG_ALIGN_GPU_H_
