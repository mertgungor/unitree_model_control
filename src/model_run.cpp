#include "model_node.cpp"

int main(int argc, char** argv)
{
  // Initialize the ROS node
  ros::init(argc, argv, "model_states_subscriber");

  // Create an instance of the ModelNode class
  ModelNode ModelNode;

  // Spin the node to receive callbacks
  ros::spin();

  return 0;
}