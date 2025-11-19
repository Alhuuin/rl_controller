#include "RL_QP_TorqueTask_Torque.h"
#include "../RLController.h"

void RL_QP_TorqueTask_Torque::configure(const mc_rtc::Configuration & config)
{
}

void RL_QP_TorqueTask_Torque::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.start_rl_state(ctl, "RL_QP_TorqueTask_Torque");
  ctl.initializeState(true, TORQUE_TASK, true);
  ctl.torqueTask->target(ctl.torque_target);
  ctl.solver().addTask(ctl.torqueTask);
  mc_rtc::log::info("RL_QP_TorqueTask_Torque state started");
}

bool RL_QP_TorqueTask_Torque::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.run_rl_state(ctl, "RL_QP_TorqueTask_Torque");
  ctl.tasksComputation(ctl.q_rl);
  ctl.torqueTask->target(ctl.torque_target);
  return false;
}

void RL_QP_TorqueTask_Torque::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController &>(ctl_);
  ctl.solver().removeTask(ctl.torqueTask);
  ctl.utils_.teardown_rl_state(ctl, "RL_QP_TorqueTask_Torque");
}

EXPORT_SINGLE_STATE("RL_QP_TorqueTask_Torque", RL_QP_TorqueTask_Torque)
