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
#include <vio/frame.h>
#include <vio/feature.h>
#include <vio/point.h>
#include <vio/config.h>
#include <vio/math_utils.h>
#include <vio/vision.h>
#include <vio/map.h>

namespace vio {

int Frame::frame_counter_ = 0;

Frame::Frame(vk::AbstractCamera* cam, const cv::Mat& img, double timestamp) :
    id_(frame_counter_++),
    timestamp_(timestamp),
    cam_(cam),
    key_pts_(5),
    is_keyframe_(false),
    v_kf_(NULL),
    T_f_w_(SE2_5(0.0,0.0,0.0))
{
  initFrame(img);
}

Frame::~Frame()
{
  for(auto&& f:fts_)f.reset();
}

void Frame::initFrame(const cv::Mat& img)
{
  // check image
  if(img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
    throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

  // Set keypoints to NULL
  for(auto&& ftr:key_pts_)ftr.reset();

  // Build Image Pyramid
   createImgPyramid(img, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);
}

void Frame::setKeyframe()
{
  is_keyframe_ = true;
  setKeyPoints();
}

void Frame::addFeature(std::shared_ptr<Feature> ftr)
{
  fts_.push_back(ftr);
}

void Frame::setKeyPoints()
{
  for(size_t i = 0; i < 5; ++i)
    if(key_pts_[i] != NULL)
      if(key_pts_[i]->point == NULL)
        key_pts_[i] = NULL;
  for(auto&& ftr:fts_)if(ftr->point != NULL) checkKeyPoints(ftr);
}

void Frame::checkKeyPoints(std::shared_ptr<Feature> ftr)
{
  const int cu = cam_->width()/2;
  const int cv = cam_->height()/2;

  // center pixel
  if(key_pts_[0] == NULL)
    key_pts_[0] = ftr;
  else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv))
        < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
    key_pts_[0] = ftr;

  if(ftr->px[0] >= cu && ftr->px[1] >= cv)
  {
    if(key_pts_[1] == NULL)
      key_pts_[1] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[1]->px[0]-cu) * (key_pts_[1]->px[1]-cv))
      key_pts_[1] = ftr;
  }
  if(ftr->px[0] >= cu && ftr->px[1] < cv)
  {
    if(key_pts_[2] == NULL)
      key_pts_[2] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[2]->px[0]-cu) * (key_pts_[2]->px[1]-cv))
      key_pts_[2] = ftr;
  }
  if(ftr->px[0] < cv && ftr->px[1] < cv)
  {
    if(key_pts_[3] == NULL)
      key_pts_[3] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[3]->px[0]-cu) * (key_pts_[3]->px[1]-cv))
      key_pts_[3] = ftr;
  }
  if(ftr->px[0] < cv && ftr->px[1] >= cv)
  {
    if(key_pts_[4] == NULL)
      key_pts_[4] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[4]->px[0]-cu) * (key_pts_[4]->px[1]-cv))
      key_pts_[4] = ftr;
  }
}

void Frame::removeKeyPoint(std::shared_ptr<Feature> ftr)
{
    bool found = false;
    if(key_pts_.size()!=5 ){
        key_pts_.clear();
        key_pts_.reserve(5);
        found=true;
    }
  for(auto&& key_pt:key_pts_){
      if(key_pt==NULL)continue;
      if(key_pt==ftr){
          key_pt=NULL;
          found=true;
      }
  }
  if(found)
    setKeyPoints();
}

bool Frame::isVisible(const Vector3d& xyz_w) const
{
  if(!id_)return false;
  if(xyz_w.hasNaN())return false;
  Vector3d xyz_f = this->se3().inverse()*xyz_w;
  if(xyz_f.z() < 0.0)
    return false; // point is behind the camera
  Vector2d px = f2c(xyz_f);
  if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
    return true;
  return false;
}

void Frame::createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  for(int i=1; i<n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
    vk::halfSample(pyr[i-1], pyr[i]);
  }
}

bool Frame::getSceneDepth(vio::Map& map,double& depth_mean, double& depth_min)
{
  vector<double> depth_vec;
  for(auto it=fts_.begin(); it!=fts_.end();)
  {
      if((*it)->point==NULL){
          ++it;
          continue;
      }
      double z=w2f((*it)->point->pos_).z();
      if((*it)->point->pos_.hasNaN() || (*it)->point->pos_.norm()==0. || z<0.05 || z > 20.0){
          map.safeDeletePoint((*it)->point);
          it = fts_.erase(it);
          continue;
      }
      depth_vec.push_back(z);
      ++it;
  }
  if(depth_vec.empty())
  {
    return false;
  }
  double max;
  depth_mean = vk::getMean(depth_vec,depth_min,max);
  return true;
}

} // namespace vio
