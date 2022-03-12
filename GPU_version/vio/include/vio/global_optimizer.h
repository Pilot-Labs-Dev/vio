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

#ifndef VIO_DEPTH_FILTER_H_
#define VIO_DEPTH_FILTER_H_

#include <queue>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <vio/global.h>
#include <vio/feature_detection.h>
#include <vio/matcher.h>
#include <vio/map.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
/*#if defined G2O_HAVE_CHOLMOD
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#elif defined G2O_HAVE_CSPARSE*/
#include <g2o/solvers/csparse/linear_solver_csparse.h>
/*#endif*/
#include <g2o/solvers/structure_only/structure_only_solver.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/stuff/sampler.h>
#include <memory>

G2O_USE_OPTIMIZATION_LIBRARY(csparse)


namespace vio {

class BA_Glob
{
public:

  BA_Glob(Map& map);

  virtual ~BA_Glob();

  /// Start this thread when seed updating should be in a parallel thread.
  void startThread();

  /// Stop the parallel thread that is running.
  void stopThread();

  void new_key_frame(){
      boost::unique_lock< boost::mutex > lk( mtx_);
      new_keyframe_=true;
      cond_.notify_one();
#if VIO_DEBUG
      fprintf(log_,"[%s] New key frame \n",
              vio::time_in_HH_MM_SS_MMM().c_str());
#endif
  }
 boost::mutex ba_mux_;
protected:
  bool new_keyframe_=false;
  boost::condition_variable cond_;
  boost::mutex mtx_;
  boost::thread* thread_;
  Map& map_;
  size_t v_id_ = 0;
  std::unique_ptr<g2o::SparseOptimizer> optimizer_=NULL;
  std::shared_ptr<g2o::CameraParameters> cam_params_=NULL;

#if VIO_DEBUG
  FILE* log_=nullptr;
#endif

/// Create a g2o vertice from a keyframe object.
        std::shared_ptr<g2o::VertexSE3Expmap> createG2oFrameSE3(
                FramePtr kf, bool state);
    /// Creates a g2o vertice from a mappoint object.
        std::shared_ptr<g2o::VertexPointXYZ> createG2oPoint(
                Vector3d pos);
  /// Creates a g2o edge between a g2o keyframe and mappoint vertice with the provided measurement.
  std::shared_ptr<g2o::EdgeProjectXYZ2UV> createG2oEdgeSE3(
          std::shared_ptr<g2o::VertexSE3Expmap> v_kf,
          std::shared_ptr<g2o::VertexPointXYZ>v_mp,
                const Vector2d& f_up,
                bool robust_kernel,
                double huber_width,
                double weight = 1);
  /// A thread that is continuously optimizing the map.
  /// Global bundle adjustment.
  void updateLoop();
  void reset_map();
};

} // namespace vio

#endif // VIO_DEPTH_FILTER_H_
