#ifndef UNITREE_CUSTOM_HPP
#define UNITREE_CUSTOM_HPP

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "model.hpp"
#include <math.h>
#include <iostream>
#include <unistd.h>


using namespace UNITREE_LEGGED_SDK;

class UnitreeCustom{
    public:
        UnitreeCustom(uint8_t level, Model model);
        void UDPSend();
        void UDPRecv();
        void RobotControl();

        Model model;
        Safety safe;
        UDP udp;
        LowCmd cmd = {0};
        LowState state = {0};
        int motiontime = 0;
        float dt = 0.002;     // 0.001~0.01

};

#endif