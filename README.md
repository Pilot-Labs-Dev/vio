VIO
===

This code implements a semi-direct monocular visual odometry pipeline which has been integrated with a EKF to infused IMU and CMD as well. The feature extraction and image alignment part are implemented in OpenCL to take advantage of GPU.

    
#### Documentation
OpenCL documentation

#### Instructions
Please be sure that you have the OpenCL driver installed


## Setting

You can edit the algorithm parameters in the follwoing file, please look at the config file to underestand the meaning of the parameters (GPU_version/vio/include/vio/config.h)

GPU_version/vio/param/vo_fast.yaml

# The dependencies:
1. OpenCV 4
2. Eigen
3. G2o  # build g2o with -DG2O_HAVE_OPENGL=ON -DBUILD_WITH_MARCH_NATIVE=ON
        # G2o version => https://github.com/RainerKuemmerle/g2o/tree/memory_management
4. Boost
5. OpenCL

# known issues
    1. The estimated oriantetion is not accurate 
    2. There is a scale map issue in the algorithm 
    3. The number of matched points are increasing a lot we need to limit them in a way that will not cause some error in the estimated odometry
    
# Log files
log files will be written in the project folder, you can change the path in the cmake files as well as activating debug mode or not
https://github.com/Pilot-Labs-Dev/vio/blob/111141365d86e3260cb75a9235afa47f0a1397fe/GPU_version/vio/CMakeLists.txt#L89

