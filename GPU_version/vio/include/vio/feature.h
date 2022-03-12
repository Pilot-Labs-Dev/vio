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

#ifndef VIO_FEATURE_H_
#define VIO_FEATURE_H_

#include <vio/frame.h>

namespace vio {

/// A salient image region that is tracked across frames.
struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType {
    CORNER,
    EDGELET
  };

  FeatureType type;     //!< Type can be corner or edgelet.
  std::shared_ptr<Frame> frame;         //!< Pointer to frame in which the feature was detected.
  Vector2d px;          //!< Coordinates in pixels on pyramid level 0.
  Vector3d f;           //!< Unit-bearing vector of the feature.
  int level;            //!< Image pyramid level where feature was extracted.
  std::shared_ptr<Point> point;         //!< Pointer to 3D point which corresponds to the feature.
  Vector2d grad;        //!< Dominant gradient direction for edglets, normalized.
  float score=0.0;
  uint8_t descriptor[64]={0}; //!< descriptor of the feature in the frame in which the feature was detected.

  Feature(std::shared_ptr<Frame> _frame, const Vector2d& _px, int _level) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    grad(1.0,0.0)
  {}

  Feature(std::shared_ptr<Frame> _frame, const Vector2d& _px, const Vector3d& _f, int _level) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    grad(1.0,0.0)
  {}

  Feature(std::shared_ptr<Frame> _frame, std::shared_ptr<Point> _point, const Vector2d& _px, const Vector3d& _f, int _level) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(_point),
    grad(1.0,0.0)
  {}
  Feature(std::shared_ptr<Frame> _frame, std::shared_ptr<Point> _point, const Vector2d& _px, const Vector3d& _f, const float _score ,int _level,uint8_t* _descriptor) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(_point),
    score(_score),
    grad(1.0,0.0)
  {
      memcpy(descriptor,_descriptor, sizeof(uint8_t)*64);
  }
  Feature(std::shared_ptr<Frame> _frame, std::shared_ptr<Point> _point, const Vector2d& _px, const float _score ,int _level,uint8_t* _descriptor) :
            type(CORNER),
            frame(_frame),
            px(_px),
            f(frame->cam_->cam2world(px)),
            level(_level),
            point(_point),
            score(_score),
            grad(1.0,0.0)
  {
        memcpy(descriptor,_descriptor, sizeof(uint8_t)*64);
  }
  Feature(std::shared_ptr<Frame> _frame, const Vector2d& _px, const float _score ,int _level,uint8_t* _descriptor) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    score(_score),
    grad(1.0,0.0)
  {
      memcpy(descriptor,_descriptor, sizeof(uint8_t)*64);
  }
};

} // namespace vio

#endif // VIO_FEATURE_H_
