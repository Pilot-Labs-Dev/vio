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

#include <vio/abstract_camera.h>
#include <stdlib.h>
#include <Eigen/StdVector>
#include <boost/bind.hpp>
#include <fstream>
#include <vio/frame_handler_base.h>
#include <vio/config.h>
#include <vio/feature.h>
#include <vio/matcher.h>
#include <vio/map.h>
#include <vio/point.h>

namespace vio
{

// definition of global and static variables which were declared in the header

FrameHandlerBase::FrameHandlerBase() :
  stage_(STAGE_PAUSED),
  set_reset_(false),
  set_start_(false),
  num_obs_last_(0)
{
}

FrameHandlerBase::~FrameHandlerBase()
{
}

bool FrameHandlerBase::startFrameProcessingCommon(const double timestamp)
{
  if(set_start_)
  {
    resetAll();
    stage_ = STAGE_FIRST_FRAME;
  }

  if(stage_ == STAGE_PAUSED)
    return false;

  map_.emptyTrash();
  return true;
}

int FrameHandlerBase::finishFrameProcessingCommon(
    const size_t update_id,
    const UpdateResult dropout,
    const size_t num_observations)
{

  if(stage_ == STAGE_DEFAULT_FRAME)
  num_obs_last_ = num_observations;

  if(dropout == RESULT_FAILURE &&
      (stage_ == STAGE_DEFAULT_FRAME || stage_ == STAGE_RELOCALIZING ))
  {
    //stage_ = STAGE_RELOCALIZING;
  }
  if(set_reset_)
    resetAll();

  return 0;
}

void FrameHandlerBase::resetCommon()
{
  map_.reset();
  stage_ = STAGE_PAUSED;
  set_reset_ = false;
  set_start_ = false;
  num_obs_last_ = 0;
}

bool ptLastOptimComparator(Point* lhs, Point* rhs)
{
  return (lhs->last_structure_optim_ < rhs->last_structure_optim_);
}

void FrameHandlerBase::optimizeStructure(
    FramePtr frame,
    size_t max_n_pts,
    int max_iter)
{
  for(auto&& it:frame->fts_){
      if(it->point==NULL)continue;
      if(it->point->obs_.size()<2){
          it->point->last_frame_overlap_id_= frame->id_;
          continue;
      }
      it->point->optimize(max_iter);
      it->point->last_frame_overlap_id_= frame->id_;
  }
}
void FrameHandlerBase::posEdit(
            FramePtr frame)
    {
        for(auto&& it:frame->fts_){
            if(it->point!=NULL && !it->point->pos_.hasNaN() && it->point->pos_.norm() !=0.){
                if(it->point->obs_.size()==2){
                    Eigen::Matrix<double,4,4> A,frame_a,frame_b;
                    frame_a=it->point->obs_.front()->frame->se3().matrix();
                    frame_b=it->point->obs_.back()->frame->se3().matrix();
                    A.row(0) = it->point->obs_.front()->f(0) * frame_a.row(2) - it->point->obs_.front()->f(2) * frame_a.row(0);
                    A.row(1) = it->point->obs_.front()->f(1) * frame_a.row(2) - it->point->obs_.front()->f(2) * frame_a.row(1);
                    A.row(2) = it->point->obs_.back()->f(0) * frame_b.row(2) - it->point->obs_.back()->f(2) * frame_b.row(0);
                    A.row(3) = it->point->obs_.back()->f(1) * frame_b.row(2) - it->point->obs_.back()->f(2) * frame_b.row(1);
                    // Aを特異値分解する (A = U S Vt)
                    // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
                    Eigen::JacobiSVD<Eigen::Matrix<double,4,4>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    const Eigen::Matrix<double,4,1> singular_vector = svd.matrixV().block<4, 1>(0, 3);
                    it->point->pos_=singular_vector.block<3, 1>(0, 0) / singular_vector(3);
                }
                //if(frame->w2f(it->point->pos_).z()<1e-9)map_.safeDeletePoint(it->point);
            }else{
                map_.safeDeletePoint(it->point);
            }
        }
    }

} // namespace vio
