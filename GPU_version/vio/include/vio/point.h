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

#ifndef VIO_POINT_H_
#define VIO_POINT_H_

#include <boost/noncopyable.hpp>
#include <vio/global.h>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <g2o/types/sba/types_six_dof_expmap.h>
/*#include <g2o/types/icp/types_icp.h>*/
static boost::mutex point_mut_;

namespace vio {

class Feature;

typedef Matrix<double, 2, 3> Matrix23d;

/// A 3D point on the surface of the scene.
class Point
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum PointType {
    TYPE_DELETED,
    TYPE_CANDIDATE,
    TYPE_UNKNOWN,
    TYPE_GOOD
  };

  static int                  point_counter_;           //!< Counts the number of created points. Used to set the unique id.
  int                         id_;                      //!< Unique ID of the point.
  Vector3d                    pos_;                     //!< 3d pos of the point in the world coordinate frame.
  list<std::shared_ptr<Feature>>              obs_;                     //!< References to keyframes which observe the point.
  size_t                      n_obs_;                   //!< Number of obervations: Keyframes AND successful reprojections in intermediate frames.
  std::shared_ptr<g2o::VertexPointXYZ>     v_pt_=NULL;                    //!< Temporary pointer to the point-vertex in g2o during bundle adjustment.
  int                         last_frame_overlap_id_;    //!< Flag for the reprojection: don't reproject a pt twice.
  PointType                   type_;                    //!< Quality of the point.
  int                         n_failed_reproj_;         //!< Number of failed reprojections. Used to assess the quality of the point.
  int                         last_structure_optim_;    //!< Timestamp of last point optimization

  Point(const Vector3d& pos);
  Point(const Vector3d& pos, std::shared_ptr<Feature> ftr);
  ~Point();

  /// Add a reference to a frame.
  void addFrameRef(std::shared_ptr<Feature> ftr);

  /// Remove reference to a frame.
  bool deleteFrameRef(FramePtr frame);


  /// Check whether mappoint has reference to a frame.
  std::shared_ptr<Feature> findFrameRef(FramePtr frame);

  /// Get Frame with similar viewpoint.
  bool getCloseViewObs(const Vector2d& pos, std::shared_ptr<Feature>& obs, int id=0) const;

  /// Get number of observations.
  inline size_t nRefs() const { return obs_.size(); }

  /// Optimize point position through minimizing the reprojection error.
  void optimize(const size_t n_iter);

  /// Jacobian of point projection on unit plane (focal length = 1) in frame (f).
    void jacobian_xyz2uv_(
            const Vector3d& p_in_f,
            const Matrix3d& R_f_w,
            Matrix23d& point_jac,
            double * cam_params,
            SE2_5 fram_t_f_w)
    {
        double fx = cam_params[0];
        double fy = cam_params[1];
        double s = cam_params[4];
        double r = cam_params[5];
        double x_n = p_in_f.x();
        double y_n = p_in_f.y();
        double z_n = p_in_f.z();
        double x_c =  fram_t_f_w.se2().translation()(0);
        double z_c = fram_t_f_w.se2().translation()(1);
        double theta = fram_t_f_w.pitch();

        double alpha = (fx*(theta/r))-(fx*((x_n*x_n)/(r*r))*theta)+((1+3*s*theta*theta)/((r*r)+1))*((fx*x_n*x_n)/(r*r));
        double beta  =               -(fx*((x_n*y_n)/(r*r))*theta)+((1+3*s*theta*theta)/((r*r)+1))*((fx*x_n*y_n)/(r*r));
        double gamma =               -(fy*((x_n*y_n)/(r*r))*theta)+((1+3*s*theta*theta)/((r*r)+1))*((fy*x_n*y_n)/(r*r));
        double lamda = (fy*(theta/r))-(fy*((y_n*y_n)/(r*r))*theta)+((1+3*s*theta*theta)/((r*r)+1))*((fy*y_n*y_n)/(r*r));

        double Xf_Xc = x_n - x_c;
        double Zf_Zc = z_n - z_c;
        double n1 = -1*sin(theta)*Xf_Xc + cos(theta)*Zf_Zc;
        double n2 = -1*cos(theta)*Xf_Xc - sin(theta)*Zf_Zc;

        point_jac(0, 0) = ((cos(theta)/z_n)*alpha)+(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*sin(theta));
        point_jac(0, 1) = ((sin(theta)/z_n)*alpha)-(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*cos(theta));
        point_jac(0, 2) = ((1/z_n)*alpha*n1)-(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*n2);

        point_jac(1, 0) = ((cos(theta)/z_n)*gamma)+(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*sin(theta));
        point_jac(1, 1) = ((sin(theta)/z_n)*gamma)-(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*cos(theta));
        point_jac(1, 2) = ((1/z_n)*gamma*n1)-(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*n2);
        //point_jac = - point_jac * R_f_w;
    }
    void jacobian_xyz2uv(
            const Vector3d& p_in_f,
            const Matrix3d& R_f_w,
            Matrix23d& point_jac)
    {
        const double z_inv = 1.0/p_in_f[2];
        const double z_inv_sq = z_inv*z_inv;
        point_jac(0, 0) = z_inv;
        point_jac(0, 1) = 0.0;
        point_jac(0, 2) = -p_in_f[0] * z_inv_sq;
        point_jac(1, 0) = 0.0;
        point_jac(1, 1) = z_inv;
        point_jac(1, 2) = -p_in_f[1] * z_inv_sq;
        point_jac = - point_jac * R_f_w;
    }

};

} // namespace vio

#endif // VIO_POINT_H_
