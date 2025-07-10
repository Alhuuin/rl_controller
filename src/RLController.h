#pragma once

#include <mc_control/mc_controller.h>
#include <mc_control/fsm/Controller.h>
// #include <mc_tasks/TorqueTask.h>
#include <mc_tasks/PostureTask.h>
#include "RLPolicyInterface.h"
#include <memory>
#include <Eigen/Dense>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "api.h"

struct RLController : public mc_control::fsm::Controller
{
  RLController(mc_rbdyn::RobotModulePtr rm, double dt, 
               const mc_rtc::Configuration & config);
  ~RLController();
  bool run() override;

  void reset(const mc_control::ControllerResetData & reset_data) override;
  
  std::unique_ptr<RLPolicyInterface> rlPolicy_;
  // std::shared_ptr<mc_tasks::TorqueTask> torqueTask_;
  std::shared_ptr<mc_tasks::PostureTask> postureTask_;
  
  std::vector<std::string> allJoints_;      // All 19 joints in mc_rtc order
  std::vector<std::string> legJoints_;      // 10 leg joints only
  
  std::vector<int> maniskillToMcRtcIdx_;    // Action reordering: Maniskill -> mc_rtc
  std::vector<int> mcRtcToManiskillIdx_;    // Observation reordering: mc_rtc -> Maniskill
  
  // Past action for observation (size 19)
  Eigen::VectorXd pastAction_;
  
  Eigen::VectorXd kp_;
  Eigen::VectorXd kd_;
  
  // threading
  std::unique_ptr<std::thread> inferenceThread_;
  std::mutex actionMutex_;
  std::mutex observationMutex_;
  std::condition_variable inferenceCondition_;
  std::atomic<bool> shouldStopInference_;
  std::atomic<bool> newObservationAvailable_;
  std::atomic<bool> newActionAvailable_;
  
  Eigen::VectorXd currentObservation_;   // Protected by observationMutex_
  Eigen::VectorXd currentAction_;        // Protected by actionMutex_
  Eigen::VectorXd latestAction_;         // current action being used by control loop
  
  bool useAsyncInference_;
  
  Eigen::VectorXd getCurrentObservation();
  void applyAction(const Eigen::VectorXd & action);
  
  // Threading
  void startInferenceThread();
  void stopInferenceThread();
  void inferenceThreadFunction();
  void updateObservationForInference();
  Eigen::VectorXd getLatestAction();
  
  // reordering methods
  Eigen::VectorXd reorderObservationToManiskill(const Eigen::VectorXd & obs);
  Eigen::VectorXd reorderActionFromManiskill(const Eigen::VectorXd & action);
  
  Eigen::VectorXd computeImpedanceTorques(const Eigen::VectorXd & desiredPos, 
                                         const Eigen::VectorXd & currentPos,
                                         const Eigen::VectorXd & currentVel);

  void supported_robots(std::vector<std::string> & out) const override 
  { 
    out = {"h1"}; 
  }

  std::unique_ptr<mc_solver::ContactConstraint> contactConstraintTest;

private:
  void initializeAllJoints();
  void initializeImpedanceGains();
}; 