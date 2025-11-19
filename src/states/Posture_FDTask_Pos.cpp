#include "Posture_FDTask_Pos.h"
#include <mc_rtc/logging.h>
#include "../RLController.h"

void Posture_FDTask_Pos::configure(const mc_rtc::Configuration & config)
{
}

void Posture_FDTask_Pos::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.initializeState(false, FD_TASK, false);
  ctl.FDTask->refAccel(ctl.refAccel);
  ctl.solver().addTask(ctl.FDTask);
  mc_rtc::log::info("Posture_FDTask_Pos state started");
}

bool Posture_FDTask_Pos::run(mc_control::fsm::Controller & ctl_)
{ 
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.tasksComputation(ctl.q_zero_vector);
  ctl.FDTask->refAccel(ctl.refAccel);
  return false;
}

void Posture_FDTask_Pos::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.solver().removeTask(ctl.FDTask);
}

EXPORT_SINGLE_STATE("Posture_FDTask_Pos", Posture_FDTask_Pos)
