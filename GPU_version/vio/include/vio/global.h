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

#ifndef VIO_GLOBAL_H_
#define VIO_GLOBAL_H_

#include <list>
#include <vector>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se2.h>
#include <sophus/se3.h>
#include <boost/shared_ptr.hpp>
#include<Eigen/StdVector>
#ifndef RPG_VIO_VIKIT_IS_VECTOR_SPECIALIZED //Guard for rpg_vikit
#define RPG_VIO_VIKIT_IS_VECTOR_SPECIALIZED
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2d)
#endif
#include <ros/console.h>
#include <boost/thread.hpp>
#include <boost/function.hpp>


namespace vio
{

    using namespace Eigen;
    using namespace Sophus;

    const double EPS = 0.0000000001;
    const double PI = 3.14159265;

    static std::string time_in_HH_MM_SS_MMM()
    {
        using namespace std::chrono;

        // get current time
        auto now = system_clock::now();

        // get number of milliseconds for the current second
        // (remainder after division into seconds)
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        // convert to std::time_t in order to convert to std::tm (broken time)
        auto timer = system_clock::to_time_t(now);

        // convert to broken time
        std::tm bt = *std::localtime(&timer);

        std::ostringstream oss;

        oss << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

        return oss.str();
    };
    class SE2_5: public SE2{
    public:
        SE2_5(SE2&& se2){
            T2_=new SE2(se2);
            assert(T2_!= nullptr);
        }
        SE2_5(SE2& se2){
            T2_=new SE2(se2);
            assert(T2_!= nullptr);
        }
        SE2_5(SE3&& se3){
            Quaterniond q=se3.unit_quaternion().normalized();
            auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);//roll,pitch,yaw
            T2_ = new SE2(SO2(euler(1)),Vector2d(se3.translation().x(),se3.translation().z()));
            assert(T2_!= nullptr);
        }
        SE2_5(SE3& se3){
            Quaterniond q=se3.unit_quaternion().normalized();
            //Camera frame z front, x right, y down -> right hands
            auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);//roll,pitch,yaw
            T2_=new SE2(SO2(euler(1)),Vector2d(se3.translation().x(),se3.translation().z()));
            assert(T2_!= nullptr);
        }
        SE2_5(double y,double z,double pitch){
            T2_=new SE2(SO2(pitch),Vector2d(y,z));
            assert(T2_!= nullptr);
        };
        SE2 se2() const{
            assert(T2_!= nullptr);
            return *T2_;
        }
/*        SE2 inverse() const{
            assert(T2_!= nullptr);
            double pitch=atan2(T2_->so2().unit_complex().imag(),T2_->so2().unit_complex().real());
            SE2 tem=SE2(pitch+M_PI,-1.0*T2_->translation());
            return tem;
        }*/
/*        SE2_5 inverse_h() const{
            assert(T2_!= nullptr);
            return SE2_5(T2_->inverse());
        }*/
        // Rotation around y
        double pitch()const{
            assert(T2_!= nullptr);
            double pitch=atan2(T2_->so2().unit_complex().imag(),T2_->so2().unit_complex().real());
            return pitch;
        }
        SE3 se3() const{
            assert(T2_!= nullptr);
            //Todo add 15 roll orientation
            //Camera frame z front, x right, y down -> right hands
            Quaterniond q;
            q = AngleAxisd(0.122173, Vector3d::UnitX()) // roll
                * AngleAxisd(pitch(), Vector3d::UnitY())// pitch
                * AngleAxisd(0.0, Vector3d::UnitZ()); //yaw
            SE3 tem=SE3(q.toRotationMatrix(),Vector3d(T2_->translation()(0), 0.0,T2_->translation()(1)));
            return tem;
        }
        bool empty() const{
            return T2_ == nullptr ? true : false;
        }

    private:
        SE2* T2_= nullptr;
    };

    class Frame;
    typedef std::shared_ptr<Frame> FramePtr;
} // namespace vio

#endif // VIO_GLOBAL_H_
