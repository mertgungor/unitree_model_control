#include "model.hpp"

ModelParams::ModelParams(
        float num_observations,
        float damping,
        float stiffness,
        float action_scale,
        float num_of_dofs,
        float lin_vel_scale,
        float ang_vel_scale,
        float dof_pos_scale,
        float dof_vel_scale,
        float clip_obs,
        float clip_actions,
        torch::Tensor torque_limits,
        torch::Tensor d_gains,
        torch::Tensor p_gains,
        torch::Tensor commands_scale,
        torch::Tensor default_dof_pos){

            this->num_observations = num_observations;
            this->clip_obs         = clip_obs;
            this->clip_actions     = clip_actions;
            this->torque_limits    = torque_limits;
            this->damping          = damping;
            this->stiffness        = stiffness;
            this->d_gains          = d_gains;
            this->p_gains          = p_gains;
            this->action_scale     = action_scale;
            this->num_of_dofs      = num_of_dofs;
            this->lin_vel_scale    = lin_vel_scale;
            this->ang_vel_scale    = ang_vel_scale;
            this->dof_pos_scale    = dof_pos_scale;
            this->dof_vel_scale    = dof_vel_scale;
            this->default_dof_pos  = default_dof_pos;
            this->commands_scale   = commands_scale;
        }
    
ModelParams::ModelParams(){
    this->num_observations = 48;
    this->clip_obs         = 100.0;
    this->clip_actions     = 100.0;
                                            //hip, thigh, calf
    this->torque_limits    = torch::tensor({20.0, 55.0, 55.0,    // front left
                                            20.0, 55.0, 55.0,    // front right
                                            20.0, 55.0, 55.0,    // rear  left
                                            20.0, 55.0, 55.0 }); // rear  right

                                            //hip, thigh, calf
    this->default_dof_pos  = torch::tensor({ 0.1000,  0.8000, -1.5000,    // front left
                                            -0.1000,  0.8000, -1.5000,    // front right
                                             0.1000,  1.0000, -1.5000,    // rear  left
                                            -0.1000,  1.0000, -1.5000 }); // rear  right                                                
    this->damping          = 1;
    this->stiffness        = 40;
    this->d_gains          = torch::ones(12)*this->damping;
    this->p_gains          = torch::ones(12)*this->stiffness;
    this->action_scale     = 0.25;
    this->num_of_dofs      = 0;
    this->lin_vel_scale    = 2.0;
    this->ang_vel_scale    = 0.25;
    this->dof_pos_scale    = 1.0;
    this->dof_vel_scale    = 0.05;
    this->commands_scale   = torch::tensor({this->lin_vel_scale, this->lin_vel_scale, this->ang_vel_scale});
}

Model::Model(std::string path, ModelParams params) {
            this->params = params;
            this->module = torch::jit::load(path);
            this->init_params();
            this->init_observations();
        }

Model::Model() {}

torch::Tensor Model::quat_rotate_inverse(torch::Tensor q, torch::Tensor v) {
    c10::IntArrayRef shape = q.sizes();
    torch::Tensor    q_w   = q.index({torch::indexing::Slice(), -1});
    torch::Tensor    q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    torch::Tensor    a     = v * (2.0 * torch::pow(q_w, 2) - 1.0).unsqueeze(-1);
    torch::Tensor    b     = torch::cross(q_vec, v, /*dim=*/-1) * q_w.unsqueeze(-1) * 2.0;
    torch::Tensor    c     = q_vec * torch::bmm(q_vec.view({shape[0], 1, 3}), v.view({shape[0], 3, 1})).squeeze(-1) * 2.0;
    return a - b + c;
}

torch::Tensor Model::compute_torques(torch::Tensor actions) {
    actions *= this->action_scale;

    torch::Tensor torques = this->p_gains * (actions + this->default_dof_pos - this->dof_pos) - this->d_gains * this->dof_vel;
    return torch::clamp(torques, -(this->torque_limits), this->torque_limits);
}

torch::Tensor Model::compute_observation(){

    printf("compute observation called\n");

    printf("lin vel size:     %d\n", ((this->quat_rotate_inverse(this->base_quat, this->lin_vel)) * this->lin_vel_scale).sizes()[1]);
    printf("ang vel size:     %d\n", ((this->quat_rotate_inverse(this->base_quat, this->ang_vel)) * this->ang_vel_scale).sizes()[1]);
    printf("gravity vec size: %d\n", (this->quat_rotate_inverse(this->base_quat, this->gravity_vec)).sizes()[1]);
    printf("commands size:    %d\n", (this->commands * this->commands_scale).sizes()[1]);
    printf("dof pos size:     %d\n", ((this->dof_pos - this->default_dof_pos) * this->dof_pos_scale).sizes()[1]);
    printf("dof vel size:     %d\n", (this->dof_vel * this->dof_vel_scale).sizes()[1]);
    printf("actions size:     %d\n\n", (this->actions).sizes()[1]);

    torch::Tensor obs = torch::cat(
        {
        (this->quat_rotate_inverse(this->base_quat, this->lin_vel)) * this->lin_vel_scale, 
        (this->quat_rotate_inverse(this->base_quat, this->ang_vel)) * this->ang_vel_scale, 
        this->quat_rotate_inverse(this->base_quat, this->gravity_vec),
        this->commands * this->commands_scale, 
        (this->dof_pos - this->default_dof_pos) * this->dof_pos_scale, 
        this->dof_vel * this->dof_vel_scale, 
        this->actions
        }, 
        1);

    obs = torch::clamp(obs, -this->clip_obs, this->clip_obs);

    printf("observation size: %d, %d\n", obs.sizes()[0], obs.sizes()[1]);

    return obs;
}

void Model::init_observations(){
    this->lin_vel           = torch::tensor({{0.0, 0.0, 0.0}});
    this->ang_vel           = torch::tensor({{0.0, 0.0, 0.0}});
    this->gravity_vec       = torch::tensor({{0.0, 0.0, -1.0}});
    this->commands          = torch::tensor({{0.0, 0.0, 0.0}});
    this->base_quat         = torch::tensor({{0.0, 0.0, 0.0, 1.0}});
    this->dof_pos           = this->default_dof_pos;
    this->dof_vel           = torch::tensor({{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }});
    this->actions           = torch::tensor({{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }});
}

void Model::update_observations(torch::Tensor lin_vel, torch::Tensor ang_vel, torch::Tensor commands, torch::Tensor base_quat, torch::Tensor dof_pos, torch::Tensor dof_vel){
    this->lin_vel = lin_vel;
    this->ang_vel = ang_vel;
    this->commands = commands;
    this->base_quat = base_quat;
    this->dof_pos = dof_pos;
    this->dof_vel = dof_vel;
    printf("updated observations\n");
}

void Model::init_params(){

    this->num_observations = this->params.num_observations;
    this->lin_vel_scale    = this->params.lin_vel_scale;  
    this->ang_vel_scale    = this->params.ang_vel_scale;  
    this->dof_pos_scale    = this->params.dof_pos_scale;  
    this->dof_vel_scale    = this->params.dof_vel_scale;  
    this->action_scale     = this->params.action_scale;   
    this->commands_scale   = this->params.commands_scale; 
    this->default_dof_pos  = this->params.default_dof_pos;
    this->torque_limits    = this->params.torque_limits;  
    this->p_gains          = this->params.p_gains;        
    this->d_gains          = this->params.d_gains;        
    this->clip_obs         = this->params.clip_obs;       
    this->clip_actions     = this->params.clip_actions;  
    this->num_of_dofs      = this->params.num_of_dofs; 

}

torch::Tensor Model::forward() {
    printf("forward called\n");
            
    torch::Tensor obs = this->compute_observation();
    printf("observations calculated\n");

    torch::Tensor action = this->module.forward({obs}).toTensor();
    printf("action calculated\n");

    this->actions = action;
    torch::Tensor clamped = torch::clamp(action, -this->clip_actions, this->clip_actions);

    printf("--------------------------\n\n");

    return clamped;
}

