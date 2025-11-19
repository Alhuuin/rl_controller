#include "RL_QP_TorqueTask_Pos.h"
#include "../RLController.h"

void RL_QP_TorqueTask_Pos::configure(const mc_rtc::Configuration & config)
{
}

void RL_QP_TorqueTask_Pos::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.start_rl_state(ctl, "RL_QP_TorqueTask_Pos");
  ctl.initializeState(false, TORQUE_TASK, true);
  ctl.torqueTask->target(ctl.torque_target);
  ctl.solver().addTask(ctl.torqueTask);
  mc_rtc::log::info("RL_QP_TorqueTask_Pos state started");
}

bool RL_QP_TorqueTask_Pos::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.run_rl_state(ctl, "RL_QP_TorqueTask_Pos");
  ctl.tasksComputation(ctl.q_rl);
  ctl.torqueTask->target(ctl.torque_target);
  return false;
}

void RL_QP_TorqueTask_Pos::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController &>(ctl_);
  ctl.solver().removeTask(ctl.torqueTask);
  ctl.utils_.teardown_rl_state(ctl, "RL_QP_TorqueTask_Pos");
}

EXPORT_SINGLE_STATE("RL_QP_TorqueTask_Pos", RL_QP_TorqueTask_Pos)
