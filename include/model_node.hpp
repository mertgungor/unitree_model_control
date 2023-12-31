#ifndef MODEL_NODE
#define MODEL_NODE

#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/Twist.h>
#include "../lib/model.cpp"
#include "unitree_legged_msgs/MotorCmd.h"

class ModelNode
{
public:

  ModelNode();
  void modelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr& msg);
  void jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg);
  void cmdvelCallback(const geometry_msgs::Twist::ConstPtr& msg);
  void runModel(const ros::TimerEvent& event);

private:

  std::string ros_namespace;

  std::vector<std::string> torque_command_topics;

  ros::Subscriber model_state_subscriber_;
  ros::Subscriber joint_state_subscriber_;
  ros::Subscriber cmd_vel_subscriber_;

  std::map<std::string, ros::Publisher> torque_publishers;
  std::vector<unitree_legged_msgs::MotorCmd> torque_commands;

  geometry_msgs::Twist vel;
  geometry_msgs::Pose pose;
  geometry_msgs::Twist cmd_vel;

  std::vector<std::string> joint_names;
  std::vector<double> joint_positions;
  std::vector<double> joint_velocities;

  Model model;

  torch::Tensor torques;

  ros::Timer timer;

  std::chrono::high_resolution_clock::time_point start_time;

};

#endif