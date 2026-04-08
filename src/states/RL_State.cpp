#include "RL_State.h"
#include "../RLController.h"

void RL_State::configure(const mc_rtc::Configuration & config)
{
}

void RL_State::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  if (!ctl.datastore().call<bool>("EF_Estimator::isActive")) {
    ctl.datastore().call("EF_Estimator::toggleActive");
  }
  ctl.utilsClass.start_rl_state(ctl, "RL_State");
  ctl.solver().addTask(ctl.torqueJointTask);
  mc_rtc::log::info("RLState started");
}

bool RL_State::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.utilsClass.run_rl_state(ctl);
  ctl.torqueJointTask->setPosTarget(ctl.q_rl);
  return false;
}

void RL_State::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.solver().removeTask(ctl.torqueJointTask);
  ctl.utilsClass.teardown_rl_state(ctl);
}

EXPORT_SINGLE_STATE("RL_State", RL_State)
