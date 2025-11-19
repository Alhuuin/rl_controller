#pragma once

#include <mc_control/fsm/State.h>

struct MC_CONTROL_FSM_STATE_DLLAPI RL_QP_TorqueTask_Pos : mc_control::fsm::State
{
  void configure(const mc_rtc::Configuration & config) override;

  void start(mc_control::fsm::Controller & ctl) override;

  bool run(mc_control::fsm::Controller & ctl) override;

  void teardown(mc_control::fsm::Controller & ctl) override;
};
