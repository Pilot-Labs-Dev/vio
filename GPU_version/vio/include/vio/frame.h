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

#ifndef VIO_FRAME_H_
#define VIO_FRAME_H_

#include <sophus/se3.h>
#include <vio/math_utils.h>
#include <vio/abstract_camera.h>
#include <boost/noncopyable.hpp>
#include <vio/global.h>
#include <g2o/types/sba/types_six_dof_expmap.h>



namespace vio {

    class Point;
    class Map;
    struct Feature;

    typedef list<std::shared_ptr<Feature>> Features;
    typedef vector<cv::Mat> ImgPyr;


/// A frame saves the image, the associated features and the estimated pose.
    class Frame
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static int                    frame_counter_;         //!< Counts the number of created frames. Used to set the unique id.
        int                           id_;                    //!< Unique id of the frame.
        double                        timestamp_;             //!< Timestamp of when the image was recorded.
        vk::AbstractCamera*           cam_;                   //!< Camera model.
        SE2_5                         T_f_w_;                 //!< Transform (f)rame from (w)orld.
        Matrix<double, 3, 3>          Cov_;                   //!< Covariance.
        ImgPyr                        img_pyr_;               //!< Image Pyramid.
        Features                      fts_;                   //!< List of features in the image.
        vector<std::shared_ptr<Feature>>  key_pts_;               //!< Five features and associated 3D points which are used to detect if two frames have overlapping field of view.
        bool                          is_keyframe_;           //!< Was this frames selected as keyframe?
        std::shared_ptr<g2o::VertexSE3Expmap>         v_kf_=NULL;                  //!< Temporary pointer to the g2o node object of the keyframe.
        int                           last_published_ts_;     //!< Timestamp of last publishing.

        Frame(vk::AbstractCamera* cam, const cv::Mat& img, double timestamp);
        ~Frame();

        /// Initialize new frame and create image pyramid.
        void initFrame(const cv::Mat& img);

        /// Select this frame as keyframe.
        void setKeyframe();

        /// Add a feature to the image
        void addFeature(std::shared_ptr<Feature> ftr);

        /// The KeyPoints are those five features which are closest to the 4 image corners
        /// and to the center and which have a 3D point assigned. These points are used
        /// to quickly check whether two frames have overlapping field of view.
        void setKeyPoints();

        /// Check if we can select five better key-points.
        void checkKeyPoints(std::shared_ptr<Feature> ftr);

        /// If a point is deleted, we must remove the corresponding key-point.
        void removeKeyPoint(std::shared_ptr<Feature> ftr);

        /// Return number of point observations.
        inline size_t nObs() const { return fts_.size(); }

        /// Check if a point in (w)orld coordinate frame is visible in the image.
        bool isVisible(const Vector3d& xyz_w) const;

        /// Full resolution image stored in the frame.
        inline const cv::Mat& img() const { return img_pyr_[0]; }

        /// Was this frame selected as keyframe?
        inline bool isKeyframe() const { return is_keyframe_; }

        /// Transforms point coordinates in world-frame (w) to camera pixel coordinates (c).
        inline Vector2d w2c(const Vector3d& xyz_w) const {
            return cam_->world2cam( T_f_w_.se3().inverse()*xyz_w);
        }
        /// Transforms point coordinates in world-frame (w) to camera pixel coordinates (c).
        inline Vector2d w2px(const Vector3d& xyz_w) const { return cam_->world2cam( w2f(xyz_w) ); }

        /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
        inline Vector3d c2f(const Vector2d& px) const { return cam_->cam2world(px[0], px[1]); }

        /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
        inline Vector3d c2f(const double x, const double y) const { return cam_->cam2world(x, y); }

        /// Transforms point coordinates in world-frame (w) to camera-frams (f).
        inline Vector3d w2f(const Vector3d& xyz_w) const {
            return Vector3d(T_f_w_.se3().inverse()*xyz_w);
        }


        /// Projects Point from unit sphere (f) in camera pixels (c).
        inline Vector2d f2c(const Vector3d& f) const { return cam_->world2cam( f ); }

        /// Return the pose of the frame in the (w)orld coordinate frame.
        inline Vector2d pos() const {
            assert(!T_f_w_.empty());
            SE2 tem=T_f_w_.se2();
            return tem.translation();
        }

        inline SE3 se3() const{
            return T_f_w_.se3();
        }


        /// Frame jacobian for projection of 3D point in (f)rame coordinate to
        /// unit plane coordinates uv (focal length = 1).
        inline static void jacobian_xyz2uv(
                const Vector3d& xyz_in_f,
                Matrix<double,2,3>& J)
        {
            const double x = xyz_in_f[0];
            const double y = xyz_in_f[1];
            const double z_inv = 1./xyz_in_f[2];
            const double z_inv_2 = z_inv*z_inv;

            J(0,0) = -z_inv;              // -1/z
            J(0,1) = x*z_inv_2;           // x/z^2
            J(0,2) = -(1.0 + pow(x,2)*z_inv_2);   // -(1.0 + x^2/z^2)

            J(1,0) = 1e-19;                // 0
            J(1,1) = y*z_inv_2;           // y/z^2
            J(1,2) = -x*y*z_inv_2;      // -x*y/z^2
        }
        void jacobian_xyz2uv_(
                const Vector3d& unit_bearing,
                const Vector3d& pos,
                Matrix<double,2,3>& J)
        {
            double fx = cam_->params()[0];
            double fy = cam_->params()[1];
            double s = cam_->params()[4];
            double r = cam_->params()[5];
            double x_n = pos.x();
            double y_n = pos.y();
            double z_n = pos.z();
            double x_c = T_f_w_.se2().translation().x();
            double z_c = T_f_w_.se2().translation().y();
            double theta = T_f_w_.pitch();

            double alpha = (fx*(theta/r))-(fx*((x_n*x_n)/(r*r))*theta)+((1+3*s*theta*theta)/((r*r)+1))*((fx*x_n*x_n)/(r*r));
            double beta  =               -(fx*((x_n*y_n)/(r*r))*theta)+((1+3*s*theta*theta)/((r*r)+1))*((fx*x_n*y_n)/(r*r));
            double gamma =               -(fy*((x_n*y_n)/(r*r))*theta)+((1+3*s*theta*theta)/((r*r)+1))*((fy*x_n*y_n)/(r*r));
            double lamda = (fy*(theta/r))-(fy*((y_n*y_n)/(r*r))*theta)+((1+3*s*theta*theta)/((r*r)+1))*((fy*y_n*y_n)/(r*r));

	        double Xf_Xc = x_n - x_c;
	        double Zf_Zc = z_n - z_c;
	        double n1 = -1*sin(theta)*Xf_Xc + cos(theta)*Zf_Zc;
	        double n2 = -1*cos(theta)*Xf_Xc - sin(theta)*Zf_Zc;

            J(0,0) = ((-1*cos(theta)/z_n)*alpha)-(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*sin(theta));
            J(0,1) = ((-1*sin(theta)/z_n)*alpha)+(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*cos(theta));
            J(0,2) = ((1/z_n)*alpha*n1)-(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*n2);

            J(1,0) = ((-1*cos(theta)/z_n)*gamma)-(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*sin(theta));
            J(1,1) = ((-1*sin(theta)/z_n)*gamma)+(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*cos(theta));
            J(1,2) = ((1/z_n)*gamma*n1)-(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*n2);

//            J(0,0) = ((-1*cos(theta)/z_n)*alpha)-((sin(theta)/z_n)*beta)-(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*n1);
//            J(0,1) = 0;
//            J(0,2) = ((sin(theta)/z_n)*alpha)-((cos(theta)/z_n)*beta)-(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*n2);
//
//            J(1,0) = ((-1*cos(theta)/z_n)*gamma)-((sin(theta)/z_n)*lamda)-(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*n1);
//            J(1,1) = 0;
//            J(1,2) = ((sin(theta)/z_n)*gamma)-((cos(theta)/z_n)*lamda)-(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*n2);
        }
        /// Creates an image pyramid of half-sampled images.
        void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr);

        /// Get the average depth of the features in the image.
        bool getSceneDepth(vio::Map& map,double& depth_mean, double& depth_min);
    };


} // namespace vio

#endif // VIO_FRAME_H_
