# build g2o with -DG2O_HAVE_OPENGL=ON -DBUILD_WITH_MARCH_NATIVE=ON
# G2o version => https://github.com/RainerKuemmerle/g2o/tree/memory_management

SET(PROJECT_NAME vio)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.3)

SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Set definitions
ADD_DEFINITIONS(-DVIO_TRACE)
ADD_DEFINITIONS(-DVIO_USE_ROS)


# Set build flags, set ARM_ARCHITECTURE environment variable on Odroid
# Set build flags. Set IS_ARM on odroid board as environment variable
SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add plain cmake packages 
FIND_PACKAGE(OpenCV 4 REQUIRED)
FIND_PACKAGE(Eigen REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system)
FIND_PACKAGE(OpenCL REQUIRED)
FIND_PACKAGE(G2O REQUIRED)
#FIND_PACKAGE(SuiteSparse)

FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp roslib cmake_modules image_transport cv_bridge tf geometry_msgs message_generation)

add_service_files(FILES
        getOdom.srv
        stop.srv
        start.srv)

generate_messages(DEPENDENCIES std_msgs)
catkin_package(
        DEPENDS Eigen OpenCV Boost
        CATKIN_DEPENDS roscpp roslib image_transport cv_bridge tf geometry_msgs message_runtime getOdom stop start
        INCLUDE_DIRS include
)
# Include dirs
INCLUDE_DIRECTORIES(
        include
        kernel
        ${Eigen_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OpenCL_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIR}
        /usr/include/suitesparse # for cholmod
)

# Set link libraries
LIST(APPEND LINK_LIBS
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${Boost_Fibers_LIBRARIES}
  ${OpenCL_LIBRARIES}
  ${catkin_LIBRARIES}
  ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_STRUCTURE_ONLY}
  ${G2O_SOLVER_PCG} ${G2O_SOLVER_CSPARSE_EXTENSION}
  ${G2O_TYPES_SBA}
  ${G2O_TYPES_SLAM3D}
  ${G2O_TYPES_ICP}
  cholmod cxsparse
)

file(GLOB SRC  CONFIGURE_DEPENDS "src/*.cpp" "include/sophus/*.cpp")



# Create VIO library
ADD_EXECUTABLE(${PROJECT_NAME} ${SRC})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${LINK_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)
ADD_DEFINITIONS(-DKERNEL_DIR=\"${PROJECT_SOURCE_DIR}/kernel\")
ADD_DEFINITIONS(-DPROJECT_DIR=\"${PROJECT_SOURCE_DIR}\")
ADD_DEFINITIONS(-DVIO_DEBUG=true)
#ADD_DEFINITIONS(-DG2O_USE_VENDORED_CERES=ON)
#ADD_DEFINITIONS(-DG2O_DELETE_IMPLICITLY_OWNED_OBJECTS=OFF)

