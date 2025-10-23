#pragma once

#include <mc_control/fsm/State.h>
#include <mc_tasks/CompliantPostureTask.h>

struct MC_CONTROL_FSM_STATE_DLLAPI Posture_TorqueTask_Pos_Compliance : mc_control::fsm::State
{
  void configure(const mc_rtc::Configuration & config) override;

  void start(mc_control::fsm::Controller & ctl) override;

  bool run(mc_control::fsm::Controller & ctl) override;

  void teardown(mc_control::fsm::Controller & ctl) override;

  // std::shared_ptr<mc_tasks::CompliantPostureTask> compPostureTask;
  std::shared_ptr<mc_tasks::PostureTask> compPostureTask;

  std::vector<std::string> activeJoints = {"left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
                                           "left_knee_joint", "left_ankle_joint",
                                           "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
                                           "right_knee_joint", "right_ankle_joint",
                                          "torso_joint"
                                          };
};
