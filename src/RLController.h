#pragma once

#include <mc_control/mc_controller.h>
#include <mc_control/fsm/Controller.h>
#include <mc_tasks/TorqueTask.h>
#include "RLPolicyInterface.h"
#include "PolicySimulatorHandling.h"
#include "utils.h"

#include <memory>
#include <Eigen/Dense>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <vector>
#include "api.h"
#include <numeric>

struct RLController : public mc_control::fsm::Controller
{
  RLController(mc_rbdyn::RobotModulePtr rm, double dt, 
               const mc_rtc::Configuration & config);
  ~RLController();
  bool run() override;

  void reset(const mc_control::ControllerResetData & reset_data) override;

  void logging();
  void initializeAllJoints();

  bool positionControl(bool run);
  bool torqueControl(bool run);

  Eigen::VectorXd getCurrentObservation();
  void applyAction(const Eigen::VectorXd & action);

  void TasksSimulation(Eigen::VectorXd & currentTargetPosition, bool highGains = false);
  
  // External force application
  void applyPelvisForce();
  
  // Threading
  void startInferenceThread();
  void stopInferenceThread();
  void inferenceThreadFunction();
  void updateObservationForInference();
  Eigen::VectorXd getLatestAction();
  
  // reordering methods
  Eigen::VectorXd reorderJointsToManiskill(const Eigen::VectorXd & obs);
  Eigen::VectorXd reorderJointsFromManiskill(const Eigen::VectorXd & action);
  

  // Tasks
  std::shared_ptr<mc_tasks::PostureTask> FSMPostureTask;
  std::shared_ptr<mc_tasks::PostureTask> FDTask;
  std::shared_ptr<mc_tasks::TorqueTask> torqueTask;

  std::map<std::string, std::vector<double>> torque_target; // Target torques for the torque task;
  int taskType = 1; // 0: use torqueTask, 1: use FDTask
  bool useQP = true;

  // Robot specific data
  std::vector<std::string> jointNames;
  size_t dofNumber_with_floatingBase = 0;
  size_t dofNumber = 0;

  // Gains
  Eigen::VectorXd kp_vector;
  Eigen::VectorXd kd_vector;
  Eigen::VectorXd high_kp_vector;
  Eigen::VectorXd high_kd_vector;

  // Options
  bool compensateExternalForces = false;
  bool compensateExternalForcesHasChanged = false;

  // External force application
  bool externalForceEnabled_ = false;
  bool pelvisForceEnabled_ = false;
  Eigen::Vector3d pelvisForce_;  // Force to apply to pelvis [Fx, Fy, Fz] in Newtons

  // Robot state 
  Eigen::VectorXd refAccel;
  Eigen::VectorXd tau_d;
  Eigen::VectorXd currentPos;
  Eigen::VectorXd currentVel;

  // For position control
  Eigen::VectorXd ddot_qp;
  Eigen::VectorXd ddot_qp_w_floatingBase;
  Eigen::VectorXd q_cmd;
  Eigen::VectorXd q_cmd_w_floatingBase;
  Eigen::VectorXd tau_cmd_after_pd;

  // For RL
  Eigen::VectorXd q_zero_vector;               // Reference joint positions (19 joints in mc_rtc order)
  Eigen::VectorXd a_before_vector;             // Last actions applied (19 joints in mc_rtc order)
  Eigen::VectorXd a_vector;                    // Action in mc_rtc order

  std::vector<std::string> notControlledJoints; // Joints that are not controlled by the RL controller: arms + torso, in that case q_rl = q_zero
  std::vector<std::string> mcRtcJointsOrder;
  
  Eigen::VectorXd a_simuOrder;
  std::vector<int> usedJoints_simuOrder; // Indices of the leg joints in the Maniskill order

  // RL policy 
  std::unique_ptr<RLPolicyInterface> rlPolicy_;
  std::unique_ptr<PolicySimulatorHandling> policySimulatorHandling_;
  
  std::chrono::steady_clock::time_point lastInferenceTime_;
  Eigen::VectorXd q_rl_vector;  // Hold target position between inference calls
  bool targetPositionValid_;               // Flag to check if we have a valid target
  static constexpr double INFERENCE_PERIOD_MS = 25.0;  // 40Hz = 25ms period

  
  // observation data - Policy specific
  Eigen::Vector3d baseAngVel; // Angular velocity of the base
  Eigen::Vector3d rpy; // Roll, Pitch, Yaw angles of the base
  Eigen::VectorXd legPos, legVel, legAction; // Leg position, velocity and action in mc_rtc order

  Eigen::Vector3d cmd_;                        // Command vector [vx, vy, yaw_rate]
  double phase_;                               // Current phase for periodic gait
  double phaseFreq_;                           // Phase frequency (1.2 Hz)
  std::chrono::steady_clock::time_point startPhase_; // Start time for phase calculation
    
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

  // State-specific data
  size_t stepCount_ = 0;
  double startTime_ = 0.0;
  
  // Timing and statistics
  bool logTiming_ = false;
  size_t timingLogInterval_ = 1000;  // Log every N steps
};