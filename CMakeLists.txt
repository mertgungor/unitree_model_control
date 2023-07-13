cmake_minimum_required(VERSION 3.0.2)
project(unitree_model_control)

set(CMAKE_INSTALL_RPATH "$ENV{HOME}/Downloads/libtorch/lib")
set(EXTRA_LIBS -pthread libunitree_legged_sdk_amd64.so lcm)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Torch REQUIRED)
find_package(unitree_legged_sdk REQUIRED)


find_package(catkin REQUIRED COMPONENTS
    controller_manager
    genmsg
    joint_state_controller
    robot_state_publisher
    roscpp
    gazebo_ros
    std_msgs
    tf
    geometry_msgs
    unitree_legged_msgs
)

find_package(gazebo REQUIRED)

catkin_package(
    CATKIN_DEPENDS 
    unitree_legged_msgs 
)


message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
else()
  set(ARCH arm64)
endif()

set(EXTRA_LIBS -pthread ${unitree_legged_sdk_LIBRARIES})

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${unitree_legged_sdk_INCLUDE_DIRS}
    ../unitree_controller/include

)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Declare a C++ library
add_library(${PROJECT_NAME}
    ../unitree_controller/src/body.cpp 
    src/model_node.cpp
)

add_library(model lib/model.cpp lib/model.hpp)
target_link_libraries(model "${TORCH_LIBRARIES}")
set_property(TARGET model PROPERTY CXX_STANDARD 14)

add_dependencies(${PROJECT_NAME} unitree_legged_msgs_gencpp)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES} ${EXTRA_LIBS} "${TORCH_LIBRARIES}"
)


add_executable(model_run src/model_run.cpp )
target_link_libraries(model_run ${catkin_LIBRARIES} "${TORCH_LIBRARIES}" model)