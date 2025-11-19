#include "RL_QP_FDTask_Torque.h"
#include "../RLController.h"

void RL_QP_FDTask_Torque::configure(const mc_rtc::Configuration & config)
{
}

void RL_QP_FDTask_Torque::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.start_rl_state(ctl, "RL_QP_FDTask_Torque");
  ctl.initializeState(true, FD_TASK, true);
  ctl.FDTask->refAccel(ctl.refAccel);
  ctl.solver().addTask(ctl.FDTask);
  mc_rtc::log::info("RL_QP_FDTask_Torque state started");
}

bool RL_QP_FDTask_Torque::run(mc_control::fsm::Controller & ctl_)
{ 
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.run_rl_state(ctl, "RL_QP_FDTask_Torque");
  ctl.tasksComputation(ctl.q_rl);
  ctl.FDTask->refAccel(ctl.refAccel);
  return false;
}

void RL_QP_FDTask_Torque::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController &>(ctl_);
  ctl.solver().removeTask(ctl.FDTask);
  ctl.utils_.teardown_rl_state(ctl, "RL_QP_FDTask_Torque");
}

EXPORT_SINGLE_STATE("RL_QP_FDTask_Torque", RL_QP_FDTask_Torque)
