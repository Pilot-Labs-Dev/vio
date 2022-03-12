/*
 * atan_camera.h
 *
 *  Created on: Aug 21, 2012
 *      Author: cforster
 *
 *  This class implements the FOV distortion model of Deverneay and Faugeras,
 *  Straight lines have to be straight, 2001.
 *
 *  The code is an implementation of the ATAN class in PTAM by Georg Klein using Eigen.
 */

#ifndef ATAN_CAMERA_H_
#define ATAN_CAMERA_H_

#include <stdlib.h>
#include <iostream>
#include <string>
#include <Eigen/Eigen>
#include <vio/abstract_camera.h>
#include <vio/math_utils.h>

namespace vk {

using namespace std;
using namespace Eigen;

class ATANCamera : public AbstractCamera {

private:
  double fx_, fy_;                      //!< focal length
  double fx_inv_, fy_inv_;              //!< inverse focal length
  double cx_, cy_;                      //!< projection center
  double s_, s_inv_;                    //!< distortion model coeff
  mutable double r_;
  double tans_;                         //!< distortion model coeff
  double tans_inv_;                     //!< distortion model coeff
  bool distortion_;                     //!< use distortion model?
  double* param_;

  //! Radial distortion transformation factor: returns ration of distorted / undistorted radius.
  inline double rtrans_factor(double r) const
  {
    if(r < 0.001 || s_ == 0.0)
      return 1.0;
    else{
        float y=2*r*sin(0.5 * (float)s_);
        float x=cos(0.5 * (float)s_);
        return (atan2(y,x) / (r * (float)s_));
        //return (s_inv_* atan(r * tans_) / r);
    }
  };

  //! Inverse radial distortion: returns un-distorted radius from distorted.
  inline double invrtrans(double r) const
  {
    if(s_ == 0.0)
      return r;
    return (tan(r * s_) * tans_inv_);
  };

public:

  ATANCamera(double width, double height, double fx, double fy, double dx, double dy, double s);

  ~ATANCamera();

  virtual Vector3d
  cam2world(const double& x, const double& y) const;

  virtual Vector3d
  cam2world(const Vector2d& px) const;

  virtual Vector2d
  world2cam(const Vector3d& xyz_c) const;

  virtual Vector2d
  world2cam(const Vector2d& uv) const;

  const Vector2d focal_length() const
  {
    return Vector2d(fx_, fy_);
  }

  virtual double errorMultiplier2() const
  {
    return fx_;
  }

  virtual double errorMultiplier() const
  {
    return 4*fx_*fy_;
  }
  virtual double* params() const{
      param_[0]=fx_;
      param_[1]=fy_;
      param_[2]=cx_;
      param_[3]=cy_;
      param_[4]=s_;
      param_[5]=r_;
     return  param_;
  }
};

} // end namespace vk

#endif /* ATAN_CAMERA_H_ */
