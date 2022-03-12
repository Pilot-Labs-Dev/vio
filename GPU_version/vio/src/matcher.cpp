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

#include <cstdlib>
#include <vio/abstract_camera.h>
#include <vio/vision.h>
#include <vio/math_utils.h>
#include <vio/patch_score.h>
#include <vio/matcher.h>
#include <vio/frame.h>
#include <vio/feature.h>
#include <vio/point.h>
#include <vio/config.h>
#include <vio/feature_alignment.h>

namespace vio {

namespace warp {

void getWarpMatrixAffine(
    const vk::AbstractCamera& cam_ref,
    const vk::AbstractCamera& cam_cur,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,
    Matrix2d& A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const int halfpatch_size = 4;
  Vector3d xyz_du_ref(cam_ref.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
  Vector3d xyz_dv_ref(cam_ref.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
  const Vector2d px_cur(cam_cur.world2cam(T_cur_ref*(f_ref)));
  const Vector2d px_du(cam_cur.world2cam(T_cur_ref*(xyz_du_ref)));
  const Vector2d px_dv(cam_cur.world2cam(T_cur_ref*(xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

int getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

bool warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int halfpatch_size,
    uint8_t* patch)
{
  if(patch==NULL || img_ref.empty())return false;
  const int patch_size = halfpatch_size*2 ;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  // Perform the warp on a larger patch.
  uint8_t* patch_ptr = patch;
  const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
  for (int y=0; y<patch_size; ++y)
  {
    for (int x=0; x<patch_size; ++x, ++patch_ptr)
    {
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
      px_patch *= (1<<search_level);
      const Vector2f px(A_ref_cur*px_patch + px_ref_pyr);
      if(patch_ptr == nullptr)continue;
      if (px[0]<1 || px[1]<1 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1){
          *patch_ptr = 0;
      }else{
          *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]);
      }
    }
  }
  return true;
}

} // namespace warp

bool depthFromTriangulation(
    const SE3& T_search_ref,
    const Vector3d& f_ref,
    const Vector3d& f_cur,
    double& depth)
{
  Matrix<double,3,2> A; A << T_search_ref.rotation_matrix() * f_ref, f_cur;
  const Matrix2d AtA = A.transpose()*A;
  if(AtA.determinant() < 0.000001)
    return false;
  const Vector2d depth2 = - AtA.inverse()*A.transpose()*T_search_ref.translation();
  depth = fabs(depth2[0]/depth2[1]);
  return true;
}

void Matcher::createPatchFromPatchWithBorder()
{
  uint8_t* ref_patch_ptr = patch_;
  for(int y=1; y<patch_size_+1; ++y, ref_patch_ptr += patch_size_)
  {
    uint8_t* ref_patch_border_ptr = patch_with_border_ + y*(patch_size_+2) + 1;
    for(int x=0; x<patch_size_; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}
cv::Scalar getMSSIM( const cv::Mat& i1, const cv::Mat& i2){
        const double C1 = 6.5025, C2 = 58.5225;
        /**************************** INITS *********************************/
        int d     = CV_32F;
        cv::Mat I1, I2;
        i1.convertTo(I1, d);           // cannot calculate on one byte large values
        i2.convertTo(I2, d);
        cv::Mat I2_2   = I2.mul(I2);        // I2^2
        cv::Mat I1_2   = I1.mul(I1);        // I1^2
        cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2
        /************************** END INITS *********************************/
        cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
        GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
        GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
        cv::Mat mu1_2   =   mu1.mul(mu1);
        cv::Mat mu2_2   =   mu2.mul(mu2);
        cv::Mat mu1_mu2 =   mu1.mul(mu2);
        cv::Mat sigma1_2, sigma2_2, sigma12;
        GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
        sigma1_2 -= mu1_2;
        GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
        sigma2_2 -= mu2_2;
        GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;
        cv::Mat t1, t2, t3;
        t1 = 2 * mu1_mu2 + C1;
        t2 = 2 * sigma12 + C2;
        t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
        t1 = mu1_2 + mu2_2 + C1;
        t2 = sigma1_2 + sigma2_2 + C2;
        t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
        cv::Mat ssim_map;
        divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
        cv::Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
        return mssim;
}

bool Matcher::findMatchDirect(
    const Point& pt,
    const Frame& cur_frame,
    Vector2d& px_cur)
{
  if(pt.last_frame_overlap_id_ !=cur_frame.id_)return true;
  if(pt.obs_.size()<2)return false;
  if(!pt.getCloseViewObs(cur_frame.pos(), ref_ftr_,cur_frame.id_))return false;
  if(ref_ftr_== nullptr)return false;
  if(ref_ftr_->frame== nullptr)return false;
  if(ref_ftr_->frame->cam_== nullptr)return false;
  Vector2i pxi=ref_ftr_->px.cast<int>();
  if(ref_ftr_->level==NULL || ref_ftr_->level > 5){
      if(!ref_ftr_->frame->cam_->isInFrame(pxi, 6))return false;
      ref_ftr_->level=0;
  }else {
      if (!ref_ftr_->frame->cam_->isInFrame(Vector2d(ref_ftr_->px / (1 << ref_ftr_->level)).cast<int>(),
                                            halfpatch_size_ + 2, ref_ftr_->level))
          return false;
  }
  if(ref_ftr_->frame->img_pyr_.empty())return false;
  if(cur_frame.img_pyr_.empty())return false;
  if(ref_ftr_->frame->img_pyr_[ref_ftr_->level].empty())return false;
  if(cur_frame.img_pyr_[ref_ftr_->level].empty())return false;
  // warp affine
  warp::getWarpMatrixAffine(
      *ref_ftr_->frame->cam_, *(cur_frame.cam_), ref_ftr_->px, ref_ftr_->f,
      (ref_ftr_->frame->se3().inverse()*pt.pos_).norm(),/*(Vector3d(ref_ftr_->frame->pos()(0),0.0,ref_ftr_->frame->pos()(1)) - pt.pos_).norm(),*/
      cur_frame.se3().inverse() * ref_ftr_->frame->se3(), ref_ftr_->level, A_cur_ref_);

  //search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);
  /// TODO paches will be mirrored while robot is rotating around it self
  if(!warp::warpAffine(A_cur_ref_, ref_ftr_->frame->img_pyr_[ref_ftr_->level], ref_ftr_->px,
                   ref_ftr_->level, ref_ftr_->level, halfpatch_size_+1, patch_with_border_))return false;
  createPatchFromPatchWithBorder();
  // px_cur should be set

  bool success = false;
  if(!warp::warpAffine(A_cur_ref_.inverse(), cur_frame.img_pyr_[ref_ftr_->level], px_cur,
                         ref_ftr_->level, ref_ftr_->level, halfpatch_size_+1, patch_with_border_cur_))return false;
  cv::Mat Patch_ref = cv::Mat(patch_size_+2, patch_size_+2, CV_8UC1, patch_with_border_);
  cv::Mat Patch_cur = cv::Mat(patch_size_+2, patch_size_+2, CV_8UC1, patch_with_border_cur_);
  cv::Scalar res = getMSSIM(Patch_cur, Patch_ref);
  if(res[0] >= 0.85)
        success = true;
  else
        success = false;
/*  debug(ref_ftr_->frame->img_pyr_[ref_ftr_->level],cur_frame.img_pyr_[ref_ftr_->level],ref_ftr_->px,
        px_cur,px_scaled ,success,patch_,patch_with_border_,(ref_ftr_->frame->se3().inverse()*pt.pos_).z());*/
  //px_cur = px_scaled;
  return success;
}

bool Matcher::findEpipolarMatchDirect(
    const Frame& ref_frame,
    const Frame& cur_frame,
    const Feature& ref_ftr,
    const double d_estimate,
    const double d_min,
    const double d_max,
    double& depth,FILE* log)
{
  if(isnan(d_min) || isnan(d_max))return false;
  SE2_5 T_cur_ref(SE2(cur_frame.T_f_w_.inverse() * ref_frame.T_f_w_.se2()));
  int zmssd_best = PatchScore::threshold();
  Vector2d uv_best;

  // Compute start and end of epipolar line in old_kf for match search, on unit plane!
  Vector2d A = vk::project2d(T_cur_ref.se3() * (ref_ftr.f*d_min));
  Vector2d B = vk::project2d(T_cur_ref.se3() * (ref_ftr.f*d_max));
  epi_dir_ = A - B;

  // Compute affine warp matrix
  warp::getWarpMatrixAffine(
      *ref_frame.cam_, *cur_frame.cam_, ref_ftr.px, ref_ftr.f,
      d_estimate, T_cur_ref.se3(), ref_ftr.level, A_cur_ref_);

  // feature pre-selection
  reject_ = false;
  if(ref_ftr.type == Feature::EDGELET && options_.epi_search_edgelet_filtering)
  {
    const Vector2d grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();
    const double cosangle = fabs(grad_cur.dot(epi_dir_.normalized()));
    if(cosangle < options_.epi_search_edgelet_max_angle) {
      reject_ = true;
      return false;
    }
  }

  search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);

  // Find length of search range on epipolar line
  Vector2d px_A(cur_frame.cam_->world2cam(A));
  Vector2d px_B(cur_frame.cam_->world2cam(B));
  epi_length_ = (px_A-px_B).norm() / (1<<search_level_);
#if VIO_DEBUG
        fprintf(log,"[%s]  epi_dir x= %f, y=%f: A:%f,%f B:%f,%f epi_length_:%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
                            epi_dir_.x(),epi_dir_.y(),A.x(),A.y(),B.x(),B.y(),epi_length_);
#endif

  // Warp reference patch at ref_level
  if(!warp::warpAffine(A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
                   ref_ftr.level, search_level_, halfpatch_size_+1, patch_with_border_))return false;
  createPatchFromPatchWithBorder();
  if(epi_length_ < 5.0)
  {
    px_cur_ = (px_A+px_B)/5.0;
    Vector2d px_scaled(px_cur_/(1<<search_level_));
    bool res;
    if(options_.align_1d)
      res = feature_alignment::align1D(
          cur_frame.img_pyr_[search_level_], (px_A-px_B).cast<float>().normalized(),
          patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
    else
      res = feature_alignment::align2D(
          cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
          options_.align_max_iter, px_scaled);
    if(res)
    {
      px_cur_ = px_scaled*(1<<search_level_);
      if(depthFromTriangulation(T_cur_ref.se3(), ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
        return true;
    }
    return false;
  }

  size_t n_steps = epi_length_/0.3; // one step per pixel
  Vector2d step = epi_dir_/n_steps;

  if(n_steps > options_.max_epi_search_steps)
  {
    printf("WARNING: skip epipolar search: %zu evaluations, px_lenght=%f, d_min=%f, d_max=%f.\n",
           n_steps, epi_length_, d_min, d_max);
    return false;
  }

  // for matching, precompute sum and sum2 of warped reference patch
  int pixel_sum = 0;
  int pixel_sum_square = 0;
  PatchScore patch_score(patch_);

  // now we sample along the epipolar line
  Vector2d uv = B-step;
  Vector2i last_checked_pxi(0,0);
  ++n_steps;
  for(size_t i=0; i<n_steps; ++i, uv+=step)
  {
    Vector2d px(cur_frame.cam_->world2cam(uv));
    Vector2i pxi(px[0]/(1<<search_level_)+0.5,
                 px[1]/(1<<search_level_)+0.5); // +0.5 to round to closest int

    if(pxi == last_checked_pxi)
      continue;
    last_checked_pxi = pxi;

    // check if the patch is full within the new frame
    if(!cur_frame.cam_->isInFrame(pxi, patch_size_, search_level_))
      continue;

    // TODO interpolation would probably be a good idea
    uint8_t* cur_patch_ptr = cur_frame.img_pyr_[search_level_].data
                             + (pxi[1]-halfpatch_size_)*cur_frame.img_pyr_[search_level_].cols
                             + (pxi[0]-halfpatch_size_);
    int zmssd = patch_score.computeScore(cur_patch_ptr, cur_frame.img_pyr_[search_level_].cols);

    if(zmssd < zmssd_best) {
      zmssd_best = zmssd;
      uv_best = uv;
    }
  }

  if(zmssd_best < PatchScore::threshold())
  {
    if(options_.subpix_refinement)
    {
      px_cur_ = cur_frame.cam_->world2cam(uv_best);
      Vector2d px_scaled(px_cur_/(1<<search_level_));
      bool res;
      if(options_.align_1d)
        res = feature_alignment::align1D(
            cur_frame.img_pyr_[search_level_], (px_A-px_B).cast<float>().normalized(),
            patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
      else
        res = feature_alignment::align2D(
            cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
            options_.align_max_iter, px_scaled);
      if(res)
      {
        px_cur_ = px_scaled*(1<<search_level_);
        if(depthFromTriangulation(T_cur_ref.se3(), ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
          return true;
      }
      return false;
    }
    px_cur_ = cur_frame.cam_->world2cam(uv_best);
    if(depthFromTriangulation(T_cur_ref.se3(), ref_ftr.f, vk::unproject2d(uv_best).normalized(), depth))
      return true;
  }
  return false;
}

} // namespace vio
