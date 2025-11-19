#include "PureRL_Torque.h"
#include "../RLController.h"

void PureRL_Torque::configure(const mc_rtc::Configuration & config)
{
}

void PureRL_Torque::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.start_rl_state(ctl, "PureRL_Torque");
  ctl.initializeState(true, PURE_RL, true);
  mc_rtc::log::info("PureRL_Torque state started");
}

bool PureRL_Torque::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.run_rl_state(ctl, "PureRL_Torque");
  ctl.tasksComputation(ctl.q_rl);
  return false;
}

void PureRL_Torque::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController &>(ctl_);
  ctl.utils_.teardown_rl_state(ctl, "PureRL_Torque");
}

EXPORT_SINGLE_STATE("PureRL_Torque", PureRL_Torque)
