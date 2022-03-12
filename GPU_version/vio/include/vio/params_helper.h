//
// Created by root on 5/24/21.
//

#ifndef VIO_PARAMS_HELPER_H
#define VIO_PARAMS_HELPER_H
/*
 * ros_params_helper.h
 *
 *  Created on: Feb 22, 2013
 *      Author: cforster
 *
 * from libpointmatcher_ros
 */

#include <string>
#include <ros/ros.h>
#if VIO_DEBUG
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#endif

namespace vk {

    inline
    bool hasParam(const std::string& name)
    {
        return ros::param::has(name);
    }

    template<typename T>
    T getParam(const std::string& name, const T& defaultValue)
    {
        T v;
        if(ros::param::get(name, v))
        {
            ROS_INFO_STREAM("Found parameter: " << name << ", value: " << v);
            return v;
        }
        else
            ROS_WARN_STREAM("Cannot find value for parameter: " << name << ", assigning default: " << defaultValue);
        return defaultValue;
    }

    template<typename T>
    T getParam(const std::string& name)
    {
        T v;
        if(ros::param::get(name, v))
        {
            ROS_INFO_STREAM("Found parameter: " << name << ", value: " << v);
            return v;
        }
        else
            ROS_ERROR_STREAM("Cannot find value for parameter: " << name);
        return T();
    }

} // namespace vk

#endif //VIO_PARAMS_HELPER_H
