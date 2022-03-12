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

#ifndef VIO_FRAME_HANDLER_BASE_H_
#define VIO_FRAME_HANDLER_BASE_H_

#include <queue>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>
#include <boost/thread.hpp>
#include <vio/global.h>
#include <vio/map.h>

namespace vk
{
class AbstractCamera;
//class PerformanceMonitor;
}

namespace vio
{

/// Base class for various VO pipelines. Manages the map and the state machine.
class FrameHandlerBase
{
public:
  enum Stage {
    STAGE_PAUSED,
    STAGE_FIRST_FRAME,
    STAGE_SECOND_FRAME,
    STAGE_DEFAULT_FRAME,
    STAGE_RELOCALIZING
  };
  enum TrackingQuality {
    TRACKING_INSUFFICIENT,
    TRACKING_BAD,
    TRACKING_GOOD
  };
  enum UpdateResult {
    RESULT_NO_KEYFRAME,
    RESULT_IS_KEYFRAME,
    RESULT_FAILURE
  };

  FrameHandlerBase();

  virtual ~FrameHandlerBase();

  /// Get the current map.
  const Map& map() const { return map_; }

  /// Will reset the map as soon as the current frame is finished processing.
  void reset() { set_reset_ = true; }

  /// Start processing.
  void start() { set_start_ = true; }

  /// Get the current stage of the algorithm.
  Stage stage() const { return stage_; }




protected:
  Stage stage_;                 //!< Current stage of the algorithm.
  bool set_reset_;              //!< Flag that the user can set. Will reset the system before the next iteration.
  bool set_start_;              //!< Flag the user can set to start the system when the next image is received.
  Map map_;                     //!< Map of keyframes created by the slam system.
  size_t num_obs_last_;                         //!< Number of observations in the previous frame.

  /// Before a frame is processed, this function is called.
  bool startFrameProcessingCommon(const double timestamp);

  /// When a frame is finished processing, this function is called.
  int finishFrameProcessingCommon(
      const size_t update_id,
      const UpdateResult dropout,
      const size_t num_observations);

  /// Reset the map and frame handler to start from scratch.
  void resetCommon();

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll() { resetCommon(); }


  /// Optimize some of the observed 3D points.
  virtual void optimizeStructure(FramePtr frame, size_t max_n_pts, int max_iter);
  virtual void posEdit(FramePtr frame);
};

} // namespace nslam

#endif // VIO_FRAME_HANDLER_BASE_H_
