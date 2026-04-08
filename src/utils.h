#pragma once
#include <mc_control/fsm/State.h>

struct utils
{  
    // RL states
    void start_rl_state(mc_control::fsm::Controller & ctl_, std::string state_name);
    void run_rl_state(mc_control::fsm::Controller & ctl_);
    void teardown_rl_state(mc_control::fsm::Controller & ctl_);

    // mc_rtc - RL policy interface
    Eigen::VectorXd getCurrentObservation(mc_control::fsm::Controller & ctl_);
    void applyAction(mc_control::fsm::Controller & ctl_, const Eigen::VectorXd & action);

    private:
        std::string state_name_;
        double syncTime_;
        double syncPhase_ = 0.0;
};