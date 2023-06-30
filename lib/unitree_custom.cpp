#include "unitree_custom.hpp"


UnitreeCustom::UnitreeCustom(uint8_t level, Model model): safe(LeggedType::A1), udp(level), model(model){
    udp.InitCmdData(cmd);
}

void UnitreeCustom::UDPRecv()
{ 
    udp.Recv();
}

void UnitreeCustom::UDPSend()
{  
    udp.Send();
}

void UnitreeCustom::RobotControl() 
{

    torch::Tensor dof_pos = torch::tensor({

        state.motorState[FL_0].q,
        state.motorState[FL_1].q,
        state.motorState[FL_2].q,

        state.motorState[FR_0].q,
        state.motorState[FR_1].q,
        state.motorState[FR_2].q,

        state.motorState[RL_0].q,
        state.motorState[RL_1].q,
        state.motorState[RL_2].q,

        state.motorState[RR_0].q,
        state.motorState[RR_1].q,
        state.motorState[RR_2].q,

        });

    torch::Tensor dof_vel = torch::tensor({
            
        state.motorState[FL_0].dq,
        state.motorState[FL_1].dq,
        state.motorState[FL_2].dq,

        state.motorState[FR_0].dq,
        state.motorState[FR_1].dq,
        state.motorState[FR_2].dq,

        state.motorState[RL_0].dq,
        state.motorState[RL_1].dq,
        state.motorState[RL_2].dq,

        state.motorState[RR_0].dq,
        state.motorState[RR_1].dq,
        state.motorState[RR_2].dq,

    });

    torch::Tensor ang_vel = torch::tensor({
            
        state.imu.gyroscope[0],
        state.imu.gyroscope[1],
        state.imu.gyroscope[2]

    });

    torch::Tensor base_quat = torch::tensor({
            
        state.imu.quaternion[1],
        state.imu.quaternion[2],
        state.imu.quaternion[3],
        state.imu.quaternion[0]

    });

    torch::Tensor lin_vel = torch::tensor({0.0, 0.0, 0.0});
    torch::Tensor commands = torch::tensor({0.0, 0.0, 0.0});


    torch::Tensor actions = model.forward();
    model.update_observations(lin_vel, ang_vel, commands, base_quat, dof_pos, dof_vel);
    torch::Tensor torques = model.compute_torques(actions[0])[0];

    udp.GetRecv(state);
 
    cmd.motorCmd[FL_0].tau = torques[0].item<float>();
    cmd.motorCmd[FL_1].tau = torques[1].item<float>();
    cmd.motorCmd[FL_2].tau = torques[2].item<float>();

    cmd.motorCmd[FR_0].tau = torques[3].item<float>();
    cmd.motorCmd[FR_1].tau = torques[4].item<float>();
    cmd.motorCmd[FR_2].tau = torques[5].item<float>();

    cmd.motorCmd[RL_0].tau = torques[6].item<float>();
    cmd.motorCmd[RL_1].tau = torques[7].item<float>();
    cmd.motorCmd[RL_2].tau = torques[8].item<float>();

    cmd.motorCmd[RR_0].tau = torques[9].item<float>();
    cmd.motorCmd[RR_1].tau = torques[10].item<float>();
    cmd.motorCmd[RR_2].tau = torques[11].item<float>();


    safe.PowerProtect(cmd, state, 1);

    udp.SetSend(cmd);
}
