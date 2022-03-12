/*
 * camera_loader.h
 *
 *  Created on: Feb 11, 2014
 *      Author: cforster
 */

#ifndef VIKIT_CAMERA_LOADER_H_
#define VIKIT_CAMERA_LOADER_H_

#include <string>
#include <vio/abstract_camera.h>
#include <vio/atan_camera.h>
#include <vio/params_helper.h>

namespace vk {
namespace camera_loader {

/// Load from ROS Namespace
bool loadFromRosNs(const std::string& ns, vk::AbstractCamera*& cam)
{
  bool res = true;
  std::string cam_model(getParam<std::string>(ns+"/cam_model"));
  if(cam_model == "ATAN")
  {
    cam = new vk::ATANCamera(
        getParam<int>(ns+"/cam_width"),
        getParam<int>(ns+"/cam_height"),
        getParam<double>(ns+"/cam_fx"),
        getParam<double>(ns+"/cam_fy"),
        getParam<double>(ns+"/cam_cx"),
        getParam<double>(ns+"/cam_cy"),
        getParam<double>(ns+"/cam_d0"));
  }
  else
  {
    cam = NULL;
    res = false;
  }
  return res;
}

} // namespace camera_loader
} // namespace vk

#endif // VIKIT_CAMERA_LOADER_H_
