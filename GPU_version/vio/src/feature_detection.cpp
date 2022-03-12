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

#include <vio/feature_detection.h>
#include <vio/feature.h>
#include <vio/vision.h>
#include <vio/for_it.hpp>

namespace vio {
namespace feature_detection {

AbstractDetector::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{
}

void AbstractDetector::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

void AbstractDetector::setExistingFeatures(const Features& fts)
{
  for(auto&& i:fts){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  };
}

void AbstractDetector::setGridOccpuancy(const Vector2d& px)
{
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    opencl* gpu_fast_,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels),gpu_fast_(gpu_fast_)
{
}

void FastDetector::detect(
    std::shared_ptr<Frame> frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    list<shared_ptr<Feature>>& fts)
    {
  std::vector<cv::KeyPoint> keypoints;
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    if(L>img_pyr.size())return;
    int scale = (1<<L);
    cv::Mat img=img_pyr.at(L);
    gpu_fast_->load(0,0,img);
    cl_int2* fast_corner=(cl_int2*)calloc(2000, sizeof(cl_int2));
    gpu_fast_->load(0,1,2000,fast_corner);
    cl_int icorner[1]={0};
    gpu_fast_->load(0,2,1,icorner);
    gpu_fast_->run(0,img.cols,img.rows);
    cl_int count[1]={0};
    gpu_fast_->read(0,2,1,count);
    if(count[0]<1 ){
        ROS_ERROR("Can not communicate with GPU");
        exit(0);
    }
    const int size=count[0];
    if(size<1 || size > 2000){
        gpu_fast_->release(0,0);
        gpu_fast_->release(0,1);
        gpu_fast_->release(0,2);
        free(fast_corner);
        return;
    }
    cl_int2* fast_corners=(cl_int2*)calloc(size, sizeof(cl_int2));
    gpu_fast_->read(0,1,size,fast_corners);
    for(uint i=0;i<size;++i)
    {
      if(fast_corners[i].x<5 || fast_corners[i].x>img_pyr[L].cols-5 || fast_corners[i].y>img_pyr[L].rows-5 || fast_corners[i].y<5){
          continue;
      }
      float score = vk::shiTomasiScore(img_pyr[L], fast_corners[i].x, fast_corners[i].y);
      keypoints.push_back(cv::KeyPoint(fast_corners[i].x*scale, fast_corners[i].y*scale, 7.f,-1,score));
    }
    gpu_fast_->release(0,0);
    gpu_fast_->release(0,1);
    gpu_fast_->release(0,2);
    free(fast_corner);
    free(fast_corners);
  }
  if(keypoints.size()<1){
      assert(0 && "GPU Driver crash try again!");
  }
  cv::Ptr<cv::xfeatures2d::FREAK> extractor = cv::xfeatures2d::FREAK::create(true, true, 22.0f, 4);
  cv::Mat descriptor;
  extractor->compute(frame->img(), keypoints, descriptor);
  for(auto&& p:_for(keypoints)){
      fts.push_back(make_shared<Feature>(frame, Vector2d(p.item.pt.x, p.item.pt.y), p.item.response ,0,descriptor.data+(p.index*64)));
  }
}

} // namespace feature_detection
} // namespace vio

