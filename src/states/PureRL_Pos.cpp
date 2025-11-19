#include "PureRL_Pos.h"
#include "../RLController.h"

void PureRL_Pos::configure(const mc_rtc::Configuration & config)
{
}

void PureRL_Pos::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.start_rl_state(ctl, "PureRL_Pos");
  ctl.initializeState(false, PURE_RL, true);
  mc_rtc::log::info("PureRL_Pos state started");
}

bool PureRL_Pos::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utils_.run_rl_state(ctl, "PureRL_Pos");
  ctl.tasksComputation(ctl.q_rl);
  return false;
}

void PureRL_Pos::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController &>(ctl_);
  ctl.utils_.teardown_rl_state(ctl, "PureRL_Pos");
}

EXPORT_SINGLE_STATE("PureRL_Pos", PureRL_Pos)
