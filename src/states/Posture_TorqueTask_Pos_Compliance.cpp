#include "Posture_TorqueTask_Pos_Compliance.h"
#include <mc_rtc/logging.h>
#include "../RLController.h"

void Posture_TorqueTask_Pos_Compliance::configure(const mc_rtc::Configuration & config)
{
}

void Posture_TorqueTask_Pos_Compliance::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.initializeState(false, TORQUE_TASK, false);

  // std::map<std::string, double> jws;
  // // for each joint of the robot if it is in activeJoints set the weight to 1.0 else set it to 0.0
  // for(const auto & jn : ctl.robot().refJointOrder())
  // {
  //   if(std::find(activeJoints.begin(), activeJoints.end(), jn) != activeJoints.end())
  //   {
  //     jws[jn] = 1.0;
  //     mc_rtc::log::info("Setting joint weight of {} to 1.0", jn);
  //   }
  //   else
  //   {
  //     jws[jn] = 0.01;
  //     mc_rtc::log::info("Setting joint weight of {} to 0.01", jn);
  //   }
  // }

  // std::map<std::string, double> jws2;
  // for(const auto & jn : ctl.robot().refJointOrder())
  // {
  //   if(std::find(activeJoints.begin(), activeJoints.end(), jn) != activeJoints.end())
  //   {
  //     jws2[jn] = 0.01;
  //     mc_rtc::log::info("Setting joint weight of {} to 0.01", jn);
  //   }
  //   else
  //   {
  //     jws2[jn] = 1.0;
  //     mc_rtc::log::info("Setting joint weight of {} to 1.0", jn);
  //   }
  // }

  ctl.torqueTask->target(ctl.torque_target);
  // ctl.torqueTask->jointWeights(jws);
  // ctl.torqueTask->selectActiveJoints(ctl.solver(), activeJoints);
  ctl.solver().addTask(ctl.torqueTask);

  // compPostureTask = std::make_shared<mc_tasks::PostureTask>(ctl.solver(), ctl.robot().robotIndex());
  // compPostureTask->reset();
  // // compPostureTask->stiffness(100.0);
  // compPostureTask->jointWeights(jws2);
  // // compPostureTask->selectActiveJoints(ctl.solver(), activeJoints
  // // compPostureTask->damping(4.0);
  // // compPostureTask->makeCompliant(false);
  // ctl.solver().addTask(compPostureTask);


  mc_rtc::log::info("Posture_TorqueTask_Pos_Compliance state started");
}

bool Posture_TorqueTask_Pos_Compliance::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  auto & robot = ctl.robots()[0];
  ctl.tasksComputation(ctl.q_zero_vector);
  ctl.torqueTask->target(ctl.torque_target);
  return false;
}

void Posture_TorqueTask_Pos_Compliance::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  ctl.solver().removeTask(ctl.torqueTask);
  // ctl.solver().removeTask(compPostureTask);
}

EXPORT_SINGLE_STATE("Posture_TorqueTask_Pos_Compliance", Posture_TorqueTask_Pos_Compliance)
