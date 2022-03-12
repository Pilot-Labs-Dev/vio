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

#ifndef VIO_FEATURE_DETECTION_H_
#define VIO_FEATURE_DETECTION_H_

#include <vio/global.h>
#include <vio/frame.h>
#include <vio/cl_class.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>

namespace vio {

/// Implementation of various feature detectors.
namespace feature_detection {

/// Temporary container used for corner detection. Features are initialized from these.
struct Corner
{
  int x;        //!< x-coordinate of corner in the image.
  int y;        //!< y-coordinate of corner in the image.
  int level;    //!< pyramid level of the corner.
  float score;  //!< shi-tomasi score of the corner.
  float angle;  //!< for gradient-features: dominant gradient angle.
  Corner(int x, int y, float score, int level, float angle) :
    x(x), y(y), level(level), score(score), angle(angle)
  {}
};
typedef vector<Corner> Corners;

/// All detectors should derive from this abstract class.
class AbstractDetector
{
public:
  AbstractDetector(
      const int img_width,
      const int img_height,
      const int cell_size,
      const int n_pyr_levels);

  virtual ~AbstractDetector() {
  };

  virtual void detect(
          std::shared_ptr<Frame> frame,
      const ImgPyr& img_pyr,
      const double detection_threshold,
      Features& fts) = 0;

  /// Flag the grid cell as occupied
  void setGridOccpuancy(const Vector2d& px);

  /// Set grid cells of existing features as occupied
  void setExistingFeatures(const Features& fts);
  void resetGrid();

protected:

  static const int border_ = 8; //!< no feature should be within 8px of border.
  int cell_size_;
  int n_pyr_levels_;
  int grid_n_cols_;
  int grid_n_rows_;
  vector<bool> grid_occupancy_;

  inline int getCellIndex(int x, int y, int level)
  {
    const int scale = (1<<level);
    return (scale*y)/cell_size_*grid_n_cols_ + (scale*x)/cell_size_;
  }
};
typedef boost::shared_ptr<AbstractDetector> DetectorPtr;

/// FAST detector by Majid Geravand.
class FastDetector : public AbstractDetector
{
public:
  FastDetector(
      const int img_width,
      const int img_height,
      const int cell_size,
      opencl* gpu_fast_,
      const int n_pyr_levels);

  virtual ~FastDetector() {}

  virtual void detect(
      std::shared_ptr<Frame> frame,
      const ImgPyr& img_pyr,
      const double detection_threshold,
      list<shared_ptr<Feature>>& fts);
  opencl* gpu_fast_;
};

} // namespace feature_detection
} // namespace vio

#endif // VIO_FEATURE_DETECTION_H_
