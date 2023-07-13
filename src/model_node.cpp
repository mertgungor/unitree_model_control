#include "../include/model_node.hpp"


ModelNode::ModelNode()
{
  // Initialize the ROS node
  ros::NodeHandle nh;
  start_time = std::chrono::high_resolution_clock::now();

  cmd_vel = geometry_msgs::Twist();

  torque_commands.resize(12);

  ros_namespace = "/a1_gazebo/";

  joint_names = {
    "FL_hip_joint"  ,
    "FL_thigh_joint",
    "FL_calf_joint" ,
    "FR_hip_joint"  ,
    "FR_thigh_joint",
    "FR_calf_joint" ,
    "RL_hip_joint"  ,
    "RL_thigh_joint",
    "RL_calf_joint" ,
    "RR_hip_joint"  ,
    "RR_thigh_joint",
    "RR_calf_joint" ,
  };

  // for(int i=0; i<12; i++){
  //   torque_command_topics.push_back(
  //       ros_namespace + joint_names[i].substr(0, joint_names[i].size() - 6) + "_controller/command"
  //   );
  // }

  for(int i=0; i<12; i++){
    torque_publishers[joint_names[i]] = nh.advertise<unitree_legged_msgs::MotorCmd>(
      ros_namespace + joint_names[i].substr(0, joint_names[i].size() - 6) + "_controller/command",
      10
    );
  }

  for(std::string topic : torque_command_topics){
    std::cout << topic << std::endl;
  }

  ModelParams a1_params;
  a1_params.num_observations = 48;
  a1_params.clip_obs         = 100.0;
  a1_params.clip_actions     = 100.0;                                             
  a1_params.damping          = 1;
  a1_params.stiffness        = 40;
  a1_params.d_gains          = torch::ones(12)*a1_params.damping;
  a1_params.p_gains          = torch::ones(12)*a1_params.stiffness;
  a1_params.action_scale     = 0.25;
  a1_params.num_of_dofs      = 12;
  a1_params.lin_vel_scale    = 2.0;
  a1_params.ang_vel_scale    = 0.25;
  a1_params.dof_pos_scale    = 1.0;
  a1_params.dof_vel_scale    = 0.05;
  a1_params.commands_scale   = torch::tensor({a1_params.lin_vel_scale, a1_params.lin_vel_scale, a1_params.ang_vel_scale});

                                              //hip, thigh, calf
  a1_params.torque_limits    = torch::tensor({{ 20.0, 55.0, 55.0,   // front left
                                                20.0, 55.0, 55.0,   // front right
                                                20.0, 55.0, 55.0,   // rear  left
                                                20.0, 55.0, 55.0 }}); // rear  right

                                                
  a1_params.default_dof_pos  = torch::tensor({{  0.1000,  0.8000, -1.5000,    
                                                -0.1000,  0.8000, -1.5000,    
                                                 0.1000,  1.0000, -1.5000,    
                                                -0.1000,  1.0000, -1.5000 }});   


  model = Model("/home/sstrostm/unitree_ros/src/unitree_model_control/models/policy_1.pt", a1_params);

  // Create a subscriber object
  model_state_subscriber_ = nh.subscribe<gazebo_msgs::ModelStates>(
      "/gazebo/model_states", 10, &ModelNode::modelStatesCallback, this);

  joint_state_subscriber_ = nh.subscribe<sensor_msgs::JointState>(
      "/a1_gazebo/joint_states", 10, &ModelNode::jointStatesCallback, this);

  cmd_vel_subscriber_ = nh.subscribe<geometry_msgs::Twist>(
      "/cmd_vel", 10, &ModelNode::cmdvelCallback, this);

  timer = nh.createTimer(ros::Duration(0.001), &ModelNode::runModel, this);
}

void ModelNode::modelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr& msg)
{

  vel  = msg->twist[2];
  pose = msg->pose[2];
//   printf("vel x: %.3f, y: %.3f, z: %.3f\n", vel.linear.x, vel.linear.y, vel.linear.z);

}

void ModelNode::cmdvelCallback(const geometry_msgs::Twist::ConstPtr& msg){
  cmd_vel = *msg;
}

void ModelNode::jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
  joint_positions = msg->position;
  joint_velocities = msg->velocity;

  // printf("%.3f\n", joint_velocities[0]);
}

void ModelNode::runModel(const ros::TimerEvent& event){
    // printf("Running model...\n");

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    torch::Tensor lin_vel     = torch::tensor({{vel.linear.x, vel.linear.y, vel.linear.z}});
    torch::Tensor ang_vel     = torch::tensor({{vel.angular.x, vel.angular.y, vel.angular.z}});
    torch::Tensor command     = torch::tensor({{cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z}});
    torch::Tensor orientation = torch::tensor({{pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w}});

    torch::Tensor joint_positions_tensor  = torch::tensor({{
        joint_positions[1],
        joint_positions[2],
        joint_positions[0],

        joint_positions[4],
        joint_positions[5],
        joint_positions[3],

        joint_positions[7],
        joint_positions[8],
        joint_positions[6],

        joint_positions[10],
        joint_positions[11],
        joint_positions[9]
    }});

    torch::Tensor joint_velocities_tensor = torch::tensor({{
        joint_velocities[1],
        joint_velocities[2],
        joint_velocities[0],

        joint_velocities[4],
        joint_velocities[5],
        joint_velocities[3],

        joint_velocities[7],
        joint_velocities[8],
        joint_velocities[6],

        joint_velocities[10],
        joint_velocities[11],
        joint_velocities[9]
    }});
    
    
    // printf("Tensors initialized...\n");

    model.update_observations(
        lin_vel               ,
        ang_vel               ,
        command               ,
        orientation           ,
        joint_positions_tensor,
        joint_velocities_tensor
        );
    
    
    torques = model.compute_torques(model.forward());

    for(int i = 0; i < 12; i++){
        std::cout << "Torques " << joint_names[i] << " : " << torques[0][i].item<double>() << std::endl;
    }

    for(int i = 0; i < 12; i++){
      torque_commands[i].tau = torques[0][i].item<double>();
      torque_commands[i].mode = 0x0A;
      // torque_commands[i].Kp = 2;
      // torque_commands[i].Kd = 0.5;


      torque_publishers[joint_names[i]].publish(torque_commands[i]);
    }

    // for(int i = 0; i < joint_command->data.size(); i++){
    //     joint_command->data[i] = torques[0][i].item<double>();
    // }

    // joint_command_pub->publish(*joint_command); 

}




