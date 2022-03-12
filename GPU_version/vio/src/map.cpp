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

#include <set>
#include <vio/map.h>
#include <vio/point.h>
#include <vio/frame.h>
#include <vio/feature.h>
#include <boost/bind.hpp>
#include <vio/for_it.hpp>

namespace vio {

Map::Map() {}

Map::~Map()
{
  reset();
}

void Map::reset()
{
  keyframes_.clear();
  emptyTrash();
}

bool Map::safeDeleteFrame(FramePtr frame)
{
  //boost::unique_lock<boost::mutex> lock(point_mut_);
  bool found = false;
  size_t position;
  for(auto&& keyframe:_for(keyframes_)){
      if(keyframe.item->id_==frame->id_){
          for(auto&& fts:keyframe.item->fts_)removePtFrameRef(keyframe.item, fts);
          found = true;
          position=keyframe.index;
          break;
      }
  }
  std::list<std::shared_ptr<Frame>>::iterator left=keyframes_.begin();
  std::advance(left,position);
  keyframes_.erase(left);

  if(found)
    return true;

  //VIO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
  return false;
}

void Map::removePtFrameRef(FramePtr frame, std::shared_ptr<Feature> ftr)
{
  if(ftr->point == NULL)
    return; // mappoint may have been deleted in a previous ref. removal
  if(ftr->point->obs_.size() <= 2)
  {
    // If the references list of mappoint has only size=2, delete mappoint
    safeDeletePoint(ftr->point);
    return;
  }
  ftr->point->deleteFrameRef(frame);  // Remove reference from map_point
  frame->removeKeyPoint(ftr); // Check if mp was keyMp in keyframe
}

void Map::safeDeletePoint(std::shared_ptr<Point> pt)
{
  boost::unique_lock<boost::mutex> lock(point_mut_);
  // Delete references to mappoints in all keyframes
  if(pt == NULL)return;
  if(pt->obs_.size()>1){
      for(auto&& ftr:pt->obs_){
          if(ftr->frame!=nullptr){
              if(ftr->frame->is_keyframe_){
                  ftr->frame->removeKeyPoint(ftr);
              }
          }
          if(ftr->point!=nullptr)ftr->point.reset();
      }
  }else{
      if(!pt->obs_.empty()){
          if(pt->obs_.front()->point!=nullptr){
              pt->obs_.back()->point.reset();
          }
      }
  }
  if(!pt->obs_.empty())pt->obs_.clear();
  // Delete mappoint
  deletePoint(pt);
}

void Map::deletePoint(std::shared_ptr<Point> pt)
{
  pt->type_ = Point::TYPE_DELETED;
  trash_points_.push_back(pt);
}

void Map::addKeyframe(FramePtr new_keyframe)
{
  keyframes_.push_back(new_keyframe);
}


FramePtr Map::getFurthestKeyframe(const Vector2d& pos)
{
  FramePtr furthest_kf;
  double maxdist = 0.0;
  for(auto it=keyframes_.begin(); it!=keyframes_.end(); )
  {
      if((*it)->T_f_w_.empty()){
          it=keyframes_.erase(it);
      }
    double dist = ((*it)->pos()-pos).norm();
    if(dist > maxdist) {
      maxdist = dist;
      furthest_kf = *it;
    }
      ++it;
  }
  return furthest_kf;
}


void Map::emptyTrash()
{
  if(trash_points_.empty())return;
  for(auto&& t:trash_points_)t.reset();
  trash_points_.clear();
}
bool Map::checkKeyFrames() {
    for(auto it=keyframes_.begin();it!=keyframes_.end();){
        if((*it)->T_f_w_.empty()){
            for(auto&& fts:(*it)->fts_)removePtFrameRef(*it, fts);
            it=keyframes_.erase(it);
        }
        ++it;
    }
    if(keyframes_.size()>0)return true;
    return false;
}


namespace map_debug {

void mapValidation(Map* map, int id)
{
  for(auto&& it:map->keyframes_)
    frameValidation(it, id);

}

void frameValidation(FramePtr frame, int id)
{
  for(auto it = frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point==NULL)
      continue;

    if((*it)->point->type_ == Point::TYPE_DELETED)
      printf("ERROR DataValidation %i: Referenced point was deleted.\n", id);

    if(!(*it)->point->findFrameRef(frame))
      printf("ERROR DataValidation %i: Frame has reference but point does not have a reference back.\n", id);

    pointValidation((*it)->point, id);
  }
  for(auto it=frame->key_pts_.begin(); it!=frame->key_pts_.end(); ++it)
    if(*it != NULL)
      if((*it)->point == NULL)
        printf("ERROR DataValidation %i: KeyPoints not correct!\n", id);
}

void pointValidation(std::shared_ptr<Point> point, int id)
{
  for(auto it=point->obs_.begin(); it!=point->obs_.end(); ++it)
  {
    bool found=false;
    for(auto it_ftr=(*it)->frame->fts_.begin(); it_ftr!=(*it)->frame->fts_.end(); ++it_ftr)
     if((*it_ftr)->point == point) {
       found=true; break;
     }
    if(!found)
      printf("ERROR DataValidation %i: Point %i has inconsistent reference in frame %i, is candidate = %i\n", id, point->id_, (*it)->frame->id_, (int) point->type_);
  }
}

void mapStatistics(Map* map)
{
  // compute average number of features which each frame observes
  size_t n_pt_obs(0);
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    n_pt_obs += (*it)->nObs();
  printf("\n\nMap Statistics: Frame avg. point obs = %f\n", (float) n_pt_obs/map->size());

  // compute average number of observations that each point has
  size_t n_frame_obs(0);
  size_t n_pts(0);
  std::set<std::shared_ptr<Point>> points;
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
  {
    for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
    {
      if((*ftr)->point == NULL)
        continue;
      if(points.insert((*ftr)->point).second) {
        ++n_pts;
        n_frame_obs += (*ftr)->point->nRefs();
      }
    }
  }
  printf("Map Statistics: Point avg. frame obs = %f\n\n", (float) n_frame_obs/n_pts);
}

} // namespace map_debug
} // namespace vio
