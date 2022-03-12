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

#include <vio/sparse_img_align_gpu.h>
#include <vio/for_it.hpp>

namespace vio {

SparseImgAlignGpu::SparseImgAlignGpu(
    int max_level, int min_level, int n_iter,
    Method method, bool verbose,opencl* residual) :
        max_level_(max_level),
        min_level_(min_level),
        residual_(residual)
{
  n_iter_ = n_iter;
  n_iter_init_ = n_iter_;
  method_ = method;
  verbose_ = verbose;
  eps_ = 1e-10;
}

size_t SparseImgAlignGpu::run(FramePtr ref_frame, FramePtr cur_frame, FILE* log)
{
  reset();
  cl_float3 ref_pos[1]={(float)ref_frame->pos()(0),(float)ref_frame->pos()(1),(float)ref_frame->T_f_w_.pitch()};
  cl_float3 cur_pos[1]={(float)cur_frame->pos()(0),(float)cur_frame->pos()(1),(float)cur_frame->T_f_w_.pitch()};
  cl_float3* features=(cl_float3*)calloc(ref_frame->fts_.size(), sizeof(cl_float3));
  cl_float2* featue_px=(cl_float2*)calloc(ref_frame->fts_.size(), sizeof(cl_float2));
  feature_counter_ = 0; // is used to compute the index of the cached jacobian
  for(auto it=ref_frame->fts_.begin();it!=ref_frame->fts_.end();++it){
        if((*it)->point == nullptr)continue;
        if((*it)->point->pos_.hasNaN())continue;
        if((*it)->point->pos_.norm()==0.)continue;
        Vector3d xyz_ref=(*it)->f*(ref_frame->se3().inverse()*(*it)->point->pos_).norm();
        features[feature_counter_].x=xyz_ref(0);
        features[feature_counter_].y=xyz_ref(1);
        features[feature_counter_].z=xyz_ref(2);
        featue_px[feature_counter_].x=(*it)->px(0);
        featue_px[feature_counter_].y=(*it)->px(1);
        ++feature_counter_;
  }
  if(!feature_counter_) // more than 10
  {
/*#if VIO_DEBUG
      fprintf(log,"[%s] residual zero points \n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif*/
      free(featue_px);
      free(features);
      return 0;
  }
  residual_->load(1,2,1,cur_pos);
  residual_->load(1,3,1,ref_pos);
  residual_->load(1,4,feature_counter_,features);
  residual_->load(1,5,feature_counter_,featue_px);
  cl_float* error_=(cl_float*)calloc(feature_counter_, sizeof(cl_float));
  cl_float* H=(cl_float*)calloc(9*feature_counter_, sizeof(cl_float));
  cl_float3* J=(cl_float3*)calloc(feature_counter_, sizeof(cl_float3));
  cl_float* chi2_=(cl_float*)calloc(feature_counter_, sizeof(cl_float));
  residual_->load(1,7,feature_counter_,error_);
  residual_->load(1,8,9*feature_counter_,H);
  residual_->load(1,9,feature_counter_,J);
  residual_->load(1,10,feature_counter_,chi2_);
  SE2 T_cur(cur_frame->T_f_w_.se2());///TODO temporary, we can remove it
  for(level_=max_level_; level_>=min_level_; --level_)
  {
      cv::Mat cur_img = cur_frame->img_pyr_.at(level_);
      cv::Mat ref_img = ref_frame->img_pyr_.at(level_);
      residual_->load(1,0,cur_img);
      residual_->load(1,1,ref_img);
      residual_->load(1,6,level_);
      mu_ = 1.0;
      optimize(T_cur);
      residual_->release(1,0);
      residual_->release(1,1);
  }
  cl_float3 pos[1]={0};
  residual_->read(1,2,1,pos);
  residual_->release(1,2);
  residual_->release(1,3);
  residual_->release(1,4);
  residual_->release(1,5);
  residual_->release(1,7);
  residual_->release(1,8);
  residual_->release(1,9);
  residual_->release(1,10);
  free(error_);
  free(H);
  free(J);
  free(chi2_);
  free(featue_px);
  free(features);
/*#if VIO_DEBUG
    fprintf(log,"[%s] residual out:%f %f %f \n",vio::time_in_HH_MM_SS_MMM().c_str(),pos[0].x,pos[0].y,pos[0].z);
#endif*/
  if(isnan(pos[0].x) || isnan(pos[0].y) || isnan(pos[0].z) || fabs(pos[0].z-cur_pos[0].z)>M_PI_2)return 1;
  cur_frame->T_f_w_ = SE2_5(pos[0].x,pos[0].y,pos[0].z);
  return 1;
}

double SparseImgAlignGpu::computeResiduals(
    bool linearize_system,
    bool compute_weight_scale)
{
    cl_float* error_=(cl_float*)calloc(feature_counter_, sizeof(cl_float));
    cl_float* H=(cl_float*)calloc(9*feature_counter_, sizeof(cl_float));
    cl_float3* J=(cl_float3*)calloc(feature_counter_, sizeof(cl_float3));
    cl_float* chi2_=(cl_float*)calloc(feature_counter_, sizeof(cl_float));
    cl_float scale=(float)scale_;
    residual_->reload(1,7,feature_counter_,error_);
    residual_->reload(1,8,9*feature_counter_,H);
    residual_->reload(1,9,feature_counter_,J);
    residual_->reload(1,10,feature_counter_,chi2_);
    residual_->load(1,11,scale);
    residual_->run(1,feature_counter_);
    cl_float* error=(cl_float*)calloc(feature_counter_, sizeof(cl_float));
    cl_float* chi=(cl_float*)calloc(feature_counter_, sizeof(cl_float));
    residual_->read(1,10,feature_counter_,chi);
    residual_->read(1,7,feature_counter_,error);
    for(int i = 1; i < feature_counter_; ++i){
        error[0] += error[i];
        chi[0] += chi[i];
    }
    scale_ = error[0]*(1.48f / feature_counter_);
    double out=chi[0]/(feature_counter_*8);
    free(error);
    free(error_);
    free(chi);
    free(H);
    free(J);
    free(chi2_);
    return out;
}

bool SparseImgAlignGpu::solve()
{
    cl_float* H=(cl_float*)calloc(9*feature_counter_, sizeof(cl_float));
    cl_float3* J=(cl_float3*)calloc(feature_counter_, sizeof(cl_float3));
    residual_->read(1,8,feature_counter_*9,H);
    residual_->read(1,9,feature_counter_,J);
    for(int i = 1; i < feature_counter_; ++i){
        H[0] += std::isnan(H[i*9]) ? 0.0 : H[i*9] ;
        H[1] += std::isnan(H[i*9+1]) ? 0.0 : H[i*9+1];
        H[2] += std::isnan(H[i*9+2]) ? 0.0 : H[i*9+2];
        H[3] += std::isnan(H[i*9+3]) ? 0.0 : H[i*9+3];
        H[4] += std::isnan(H[i*9+4]) ? 0.0 : H[i*9+4];
        H[5] += std::isnan(H[i*9+5]) ? 0.0 : H[i*9+5];
        H[6] += std::isnan(H[i*9+6]) ? 0.0 : H[i*9+6];
        H[7] += std::isnan(H[i*9+7]) ? 0.0 : H[i*9+7];
        H[8] += std::isnan(H[i*9+8]) ? 0.0 : H[i*9+8];
        J[0].x -= std::isnan(J[i].x) ? 0.0 : J[i].x;
        J[0].y -= std::isnan(J[i].y) ? 0.0 : J[i].y;
        J[0].z -= std::isnan(J[i].z) ? 0.0 : J[i].z;
    }
    double Hd[9]={(double)H[0],(double)H[1],(double)H[2],
                  (double)H[3],(double)H[4],(double)H[5],
                  (double)H[6],(double)H[7],(double)H[8]};
    x_ = Eigen::Matrix<double,3,3>(Hd).ldlt().solve(Eigen::Vector3d((double)J[0].x,(double)J[0].y,(double)J[0].z));
    double norm=x_.norm();
    if(norm<=0 ||norm > 1.0)x_=Eigen::Vector3d(0.1,0.1,0.1);
    free(H);
    free(J);
    return true;
}
void SparseImgAlignGpu::update()
{
    cl_float3 pos[1]={0.0,0.0,0.0};
    residual_->read(1,2,1,pos);
    Sophus::SE2 update =  Sophus::SE2(pos[0].z,Eigen::Vector2d(pos[0].x,pos[0].y)) * Sophus::SE2::exp(-1.0*x_);
    pos[0].x=(float)update.translation()(0);
    pos[0].y=(float)update.translation()(1);
    pos[0].z=(float)atan2(update.so2().unit_complex().imag(),update.so2().unit_complex().real());
    residual_->reload(1,2,1,pos);
}

void SparseImgAlignGpu::startIteration()
{}

void SparseImgAlignGpu::finishIteration()
{
}

} // namespace vio

