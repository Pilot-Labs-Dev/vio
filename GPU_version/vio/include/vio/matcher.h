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

#ifndef VIO_MATCHER_H_
#define VIO_MATCHER_H_

#include <vio/global.h>

namespace vk {
  class AbstractCamera;
  namespace patch_score {
    template<int HALF_PATCH_SIZE> class ZMSSD;
  }
}

namespace vio {

class Point;
class Frame;
class Feature;

/// Warp a patch from the reference view to the current view.
namespace warp {

void getWarpMatrixAffine(
    const vk::AbstractCamera& cam_ref,
    const vk::AbstractCamera& cam_cur,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,
    Matrix2d& A_cur_ref);

int getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level);

bool warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int level_cur,
    const int halfpatch_size,
    uint8_t* patch);

} // namespace warp

/// Patch-matcher for reprojection-matching and epipolar search in triangulation.
class Matcher
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static const int halfpatch_size_ = 4;
  static const int patch_size_ = 8;

  typedef vk::patch_score::ZMSSD<halfpatch_size_> PatchScore;

  struct Options
  {
    bool align_1d;              //!< in epipolar search: align patch 1D along epipolar line
    int align_max_iter;         //!< number of iterations for aligning the feature patches in gauss newton
    double max_epi_length_optim;//!< max length of epipolar line to skip epipolar search and directly go to img align
    size_t max_epi_search_steps;//!< max number of evaluations along epipolar line
    bool subpix_refinement;     //!< do gauss newton feature patch alignment after epipolar search
    bool epi_search_edgelet_filtering;
    double epi_search_edgelet_max_angle;
    Options() :
      align_1d(false),
      align_max_iter(10),
      max_epi_length_optim(2.0),
      max_epi_search_steps(1000),
      subpix_refinement(true),
      epi_search_edgelet_filtering(true),
      epi_search_edgelet_max_angle(0.3)
    {}
  } options_;

  uint8_t patch_[patch_size_*patch_size_] __attribute__ ((aligned (16)));
  uint8_t patch_with_border_[(patch_size_+2)*(patch_size_+2)] __attribute__ ((aligned (16)));
  uint8_t patch_with_border_cur_[(patch_size_+2)*(patch_size_+2)] __attribute__ ((aligned (16)));
  Matrix2d A_cur_ref_;          //!< affine warp matrix
  Vector2d epi_dir_;
  double epi_length_;           //!< length of epipolar line segment in pixels (only used for epipolar search)
  double h_inv_;                //!< hessian of 1d image alignment along epipolar line
  int search_level_;
  bool reject_;
  std::shared_ptr<Feature> ref_ftr_;
  Vector2d px_cur_;

  Matcher() = default;
  ~Matcher() = default;

  /// Find a match by directly applying subpix refinement.
  /// IMPORTANT! This function assumes that px_cur is already set to an estimate that is within ~2-3 pixel of the final result!
  bool findMatchDirect(
       const Point& pt,
       const Frame& frame,
      Vector2d& px_cur);

  /// Find a match by searching along the epipolar line without using any features.
  bool findEpipolarMatchDirect(
      const Frame& ref_frame,
      const Frame& cur_frame,
      const Feature& ref_ftr,
      const double d_estimate,
      const double d_min,
      const double d_max,
      double& depth,FILE* log);

  void createPatchFromPatchWithBorder();
  uint i=0;
  void debug(cv::Mat ref,cv::Mat cur, Vector2d ref_px,Vector2d cur_px_in,Vector2d cur_px_out,bool res,uint8_t* patch,uint8_t* patch_border, double depth){
      cv::Mat Ref,Cur,Patch,Patch_b;
      cv::cvtColor(cv::Mat(patch_size_,patch_size_,CV_8UC1,patch),Patch,cv::COLOR_GRAY2RGB);
      cv::cvtColor(cv::Mat(patch_size_+2,patch_size_+2,CV_8UC1,patch_border),Patch_b,cv::COLOR_GRAY2RGB);
      cv::cvtColor(cur,Cur,cv::COLOR_GRAY2RGB);
      cv::cvtColor(ref,Ref,cv::COLOR_GRAY2RGB);
      cv::circle(Ref, cv::Point(ref_px.x(),ref_px.y() ), 2, cv::Scalar(0, 0,255),2);
      cv::circle(Cur, cv::Point(cur_px_in.x(),cur_px_in.y() ), 2, cv::Scalar(0, 0,255),2);
      cv::circle(Cur, cv::Point(cur_px_out.x(),cur_px_out.y() ), 2, cv::Scalar(0, 255,0),2);
      cv::putText(Cur,res? "Accept": "reject",cv::Point(10,30),cv::FONT_HERSHEY_COMPLEX_SMALL,1.0,cv::Scalar(0,255,0),1,cv:: LINE_AA);
      cv::putText(Cur,std::to_string(depth),cv::Point(100,30),cv::FONT_HERSHEY_COMPLEX_SMALL,1.0,cv::Scalar(0,255,0),1,cv:: LINE_AA);
      cv::imwrite(std::string(PROJECT_DIR)+"/Ref"+std::to_string(i)+".png",Ref);
      cv::imwrite(std::string(PROJECT_DIR)+"/Cur"+std::to_string(i)+".png",Cur);
      cv::imwrite(std::string(PROJECT_DIR)+"/patch"+std::to_string(i)+".png",Patch);
      cv::imwrite(std::string(PROJECT_DIR)+"/patch_b"+std::to_string(i)+".png",Patch_b);
      ++i;
  }
};

} // namespace vio

#endif // VIO_MATCHER_H_
