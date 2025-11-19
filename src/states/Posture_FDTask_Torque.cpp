#include "Posture_FDTask_Torque.h"
#include <mc_rtc/logging.h>
#include "../RLController.h"

void Posture_FDTask_Torque::configure(const mc_rtc::Configuration & config)
{
}

void Posture_FDTask_Torque::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.initializeState(true, FD_TASK, false);
  ctl.FDTask->refAccel(ctl.refAccel);
  ctl.solver().addTask(ctl.FDTask);
  mc_rtc::log::info("Posture_FDTask_Torque state started");
}

bool Posture_FDTask_Torque::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);

  ctl.tasksComputation(ctl.q_zero_vector);
  ctl.FDTask->refAccel(ctl.refAccel);

  return false;
}

void Posture_FDTask_Torque::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.solver().removeTask(ctl.FDTask);
}

EXPORT_SINGLE_STATE("Posture_FDTask_Torque", Posture_FDTask_Torque)
