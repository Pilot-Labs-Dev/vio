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

#include <stdexcept>
#include <vio/pose_optimizer.h>
#include <vio/frame.h>
#include <vio/feature.h>
#include <vio/point.h>
#include <vio/robust_cost.h>
#include <vio/math_utils.h>
#include <sophus/se2.h>
#include <vio/config.h>

namespace vio {
namespace pose_optimizer {

void optimizeGaussNewton(
    const size_t n_iter,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs,
    vio::Map& map,
    FILE* log)
{
  // init
  double chi2(0.0);
  vector<double> chi2_vec_init, chi2_vec_final;
  vk::robust_cost::HuberWeightFunction weight_function;
  SE2_5 T_old(frame->T_f_w_.se2());
  Matrix3d A;
  Vector3d b;

  // compute the scale of the error for robust estimation
  std::vector<float> errors;
  for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); /*++it*/)
  {
      if((*it)->point == NULL) {
          it++;// = frame->fts_.erase(it);
          continue;
      }
      double z=frame->w2f((*it)->point->pos_).z();
      if((*it)->point->pos_.hasNaN() || (*it)->point->pos_.norm()==0. || z<0.05 || z > 20.0){
          map.safeDeletePoint((*it)->point);
          it = frame->fts_.erase(it);
          continue;
      }
      if((*it)->point->type_==vio::Point::TYPE_UNKNOWN){
          it++;
          continue;
      }
    //Reprojection error
    Vector2d e = vk::project2d((*it)->f)
               - vk::project2d(Vector3d(frame->se3()*(*it)->point->pos_));
    if(std::isnan(e.norm())){
        map.safeDeletePoint((*it)->point);
        it = frame->fts_.erase(it);
        continue;
    }
    e *= 1.0 / (1<<(*it)->level);
    chi2_vec_init.push_back(e.norm()); // just for debug
    errors.push_back(e.norm());
    it++;
  }

  if(errors.empty())
    return;
  vk::robust_cost::MADScaleEstimator scale_estimator;
  estimated_scale = scale_estimator.compute(errors);

  double scale = estimated_scale;
  for(size_t iter=0; iter<n_iter; iter++)
  {

    b.setZero();
    A.setZero();
    double new_chi2(0.0);
    // overwrite scale
    if(iter == 5)
        scale = 0.85/frame->cam_->errorMultiplier2();
    // compute residual
    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
      if((*it)->point==NULL)continue;
      if((*it)->point->type_==vio::Point::TYPE_UNKNOWN)continue;
      Matrix23d J;
      frame->jacobian_xyz2uv_((*it)->f,(*it)->point->pos_,J);
      Vector2d e = vk::project2d((*it)->f) - vk::project2d(frame->w2f((*it)->point->pos_));
      double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
      e *= sqrt_inv_cov;
      J *= sqrt_inv_cov;
      double weight = weight_function.value(e.norm()/scale);
      A.noalias() += J.transpose()*J*weight;
      b.noalias() -= J.transpose()*e*weight;
      new_chi2 += e.norm();
    }

    // solve linear system
    Vector3d dT(A.ldlt().solve(b));

    // check if error increased
    if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
    {

      frame->T_f_w_ = T_old; // roll-back
      break;
    }
    dT *=new_chi2;
    // update the model
    T_old = frame->T_f_w_;
    frame->T_f_w_=SE2_5(T_old.se2().translation().x()+dT.x(),T_old.se2().translation().y()+dT.y(),T_old.pitch()+dT.z());
    chi2 = new_chi2;

    // stop when converged
    if(vk::norm_max(dT) <= EPS)
      break;
  }
  num_obs=0;
  for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); /*++it*/)
  {
    if((*it)->point == NULL) {
        it++;// = frame->fts_.erase(it);
        continue;
    }
    if((*it)->point->pos_.hasNaN() || (*it)->point->pos_.norm()==0.){
          map.safeDeletePoint((*it)->point);
          it = frame->fts_.erase(it);
          continue;
    }
    Vector2d e = vk::project2d((*it)->f) - vk::project2d(frame->w2f((*it)->point->pos_));
    e /= (1<<(*it)->level);
    chi2_vec_final.push_back(e.norm());
    if(e.norm() >  vio::Config::poseOptimThresh() / frame->cam_->errorMultiplier2())
    {
      map.safeDeletePoint((*it)->point);
      it = frame->fts_.erase(it);
    }else{
        (*it)->point->type_=vio::Point::TYPE_CANDIDATE;
        ++num_obs;
        it++;
    }

  }
#if VIO_DEBUG
    error_init=0.0;
    error_final=0.0;
    if(!chi2_vec_init.empty())for(auto&& i:chi2_vec_init)error_init+=i;
        error_init /=chi2_vec_init.size();
    if(!chi2_vec_final.empty())for(auto&& i:chi2_vec_final)error_final+=i;
        error_final /= chi2_vec_final.size();
    fprintf(log,"[%s]  n obs with reprojection error less than 1.0 / frame->cam_->errorMultiplier2() =%d \t error init =%f \t error end=%f\n",
            vio::time_in_HH_MM_SS_MMM().c_str(),num_obs,error_init,error_final);
#endif
}

} // namespace pose_optimizer
} // namespace vio
