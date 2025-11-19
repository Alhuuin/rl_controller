#include "Posture_TorqueTask_Torque.h"
#include <mc_rtc/logging.h>
#include "../RLController.h"

void Posture_TorqueTask_Torque::configure(const mc_rtc::Configuration & config)
{
}

void Posture_TorqueTask_Torque::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.initializeState(true, TORQUE_TASK, false);
  ctl.torqueTask->target(ctl.torque_target);
  ctl.solver().addTask(ctl.torqueTask);
  mc_rtc::log::info("Posture_TorqueTask_Torque state started");
}

bool Posture_TorqueTask_Torque::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.tasksComputation(ctl.q_zero_vector);
  ctl.torqueTask->target(ctl.torque_target);
  return false;
}

void Posture_TorqueTask_Torque::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.solver().removeTask(ctl.torqueTask);
}

EXPORT_SINGLE_STATE("Posture_TorqueTask_Torque", Posture_TorqueTask_Torque)
