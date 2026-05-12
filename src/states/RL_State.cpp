#include "RL_State.h"
#include "../RLController.h"

void RL_State::configure(const mc_rtc::Configuration & config)
{
}

void RL_State::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.start_rl_state(ctl, "RL_State");
  ctl.initializeState();
  ctl.torqueTask->target(ctl.torque_target);
  ctl.solver().addTask(ctl.torqueTask);
  mc_rtc::log::info("RLState started");
}

bool RL_State::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.run_rl_state(ctl, "RL_State");
  ctl.tasksComputation(ctl.q_rl);
  ctl.torqueTask->target(ctl.torque_target);
  return false;
}

void RL_State::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.solver().removeTask(ctl.torqueTask);
  ctl.utils_.teardown_rl_state(ctl, "RL_State");
}

EXPORT_SINGLE_STATE("RL_State", RL_State)
