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

#ifndef VIO_MAP_H_
#define VIO_MAP_H_

#include <queue>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <vio/global.h>
#include <vio/point.h>
#include <vio/frame.h>
#include <vio/feature.h>

namespace vio {

/// Map object which saves all keyframes which are in a map.
class Map
{
public:
  list< FramePtr > keyframes_;          //!< List of keyframes in the map.
  list< std::shared_ptr<Point> > trash_points_;         //!< A deleted point is moved to the trash bin. Now and then this is cleaned. One reason is that the visualizer must remove the points also.

  Map();
  ~Map();

  /// Reset the map. Delete all keyframes and reset the frame and point counters.
  void reset();

  /// Delete a point in the map and remove all references in keyframes to it.
  void safeDeletePoint(std::shared_ptr<Point> pt);

  /// Moves the point to the trash queue which is cleaned now and then.
  void deletePoint(std::shared_ptr<Point> pt);

  /// Moves the frame to the trash queue which is cleaned now and then.
  bool safeDeleteFrame(FramePtr frame);

  /// Remove the references between a point and a frame.
  void removePtFrameRef(FramePtr frame, std::shared_ptr<Feature> ftr);

  /// Add a new keyframe to the map.
  void addKeyframe(FramePtr new_keyframe);

  /// Given a frame, return all keyframes which have an overlapping field of view.
  void getCloseKeyframes(const FramePtr& frame, list< pair<FramePtr,double> >& close_kfs){
      for(auto kf : keyframes_)
      {
          // check if kf has overlaping field of view with frame, use therefore KeyPoints
          for(auto keypoint : kf->key_pts_)
          {
              if(keypoint == nullptr)
                  continue;
              if(keypoint->point==NULL)continue;
              if(frame->isVisible(keypoint->point->pos_))
              {
                  close_kfs.push_back(
                          std::make_pair(
                                  kf, (frame->T_f_w_.se2().translation()-kf->T_f_w_.se2().translation()).norm()));
                  break; // this keyframe has an overlapping field of view -> add to close_kfs
              }
          }
      }
  }

  /// Return the keyframe which is furthest apart from pos.
  FramePtr getFurthestKeyframe(const Vector2d& pos);

  /// Empty trash bin of deleted keyframes and map points. We don't delete the
  /// points immediately to ensure proper cleanup and to provide the visualizer
  /// a list of objects which must be removed.
  void emptyTrash();

  bool checkKeyFrames();


  /// Return the number of keyframes in the map
  inline size_t size() const { return keyframes_.size(); }
};

/// A collection of debug functions to check the data consistency.
namespace map_debug {

void frameValidation(FramePtr frame, int id);
void pointValidation(std::shared_ptr<Point> point, int id);

} // namespace map_debug
} // namespace vio

#endif // VIO_MAP_H_
