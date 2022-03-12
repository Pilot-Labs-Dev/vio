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

#ifndef VIO_REPROJECTION_H_
#define VIO_REPROJECTION_H_

#include <vio/global.h>
#include <vio/matcher.h>
#include <CL/cl.h>
#include <vio/cl_class.h>
#include <vio/initialization.h>
#include <vio/vision.h>
#include <vio/map.h>
#include <vio/point.h>

namespace vk {
class AbstractCamera;
}

namespace vio {

/// Project points from the map into the image and find the corresponding
/// feature (corner). We don't search a match for every point but only for one
/// point per cell. Thereby, we achieve a homogeneously distributed set of
/// matched features and at the same time we can save processing time by not
/// projecting all points.
class Reprojector
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Reprojector config parameters
  struct Options {
    size_t max_n_kfs;   //!< max number of keyframes to reproject from
    bool find_match_direct;
    Options()
    : max_n_kfs(30),
      find_match_direct(true)
    {}
  } options_;

  size_t n_matches_;
  size_t n_trials_;

  Reprojector(vk::AbstractCamera* cam, Map& map);

  ~Reprojector();

  /// Project points from the map into the image. First finds keyframes with
  /// overlapping field of view and projects only those map-points.
  void reprojectMap(
      FramePtr frame,
      FramePtr last_frame,
      std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs,
      opencl* gpu_fast_,
      FILE* log_);


private:

  /// A candidate is a point that projects into the image plane and for which we
  /// will search a maching feature in the image.
  struct Candidate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector2d px;     //!< projected 2D pixel location.
    Candidate(Vector2d& px) : px(px) {}
  };
  typedef std::list<Candidate > Cell;
  typedef std::vector<Cell*> CandidateGrid;

  /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
  struct Grid
  {
    CandidateGrid cells;
    vector<int> cell_order;
    int cell_size;
    int grid_n_cols;
    int grid_n_rows;
  };

  Grid grid_;
  Matcher matcher_;
  Map& map_;

  void initializeGrid(vk::AbstractCamera* cam);
  void resetGrid();
};

} // namespace vio

#endif // VIO_REPROJECTION_H_
