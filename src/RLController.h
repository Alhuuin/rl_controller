#pragma once

#include <mc_control/mc_controller.h>
#include <mc_control/fsm/Controller.h>
// #include <mc_tasks/TorqueTask.h>
#include "RLPolicyInterface.h"
#include "RL_utils.h"

#include <memory>
#include <Eigen/Dense>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <vector>
#include "api.h"

struct RLController : public mc_control::fsm::Controller
{
  RLController(mc_rbdyn::RobotModulePtr rm, double dt, 
               const mc_rtc::Configuration & config);
  ~RLController();
  bool run() override;

  void reset(const mc_control::ControllerResetData & reset_data) override;

  void torqueTaskSimulation(Eigen::VectorXd & currentTargetPosition);

  std::shared_ptr<mc_tasks::PostureTask> FSMPostureTask;
  std::shared_ptr<mc_tasks::PostureTask> similiTorqueTask;
  // std::shared_ptr<mc_tasks::TorqueTask> torqueTask_;

  Eigen::VectorXd refAccel;
  // Eigen::VectorXd refAccel_w_floatingBase;
  // Eigen::VectorXd refPos;
  Eigen::VectorXd tau_d;
  // Eigen::VectorXd tau_d_w_floatingBase;
  Eigen::VectorXd currentPos;
  // Eigen::VectorXd currentPos_w_floatingBase;
  Eigen::VectorXd currentVel;
  // Eigen::VectorXd currentVel_w_floatingBase;

  std::vector<std::vector<double>> desiredPosture;
  std::vector<std::string> jointNames;
  std::map<std::string, std::vector<double>> postureTarget;

  std::map<std::string, double> kp;
  std::map<std::string, double> kd;
  Eigen::VectorXd kp_vector;
  Eigen::VectorXd kd_vector;

  std::map<std::string, double> high_kp;
  std::map<std::string, double> high_kd;
  Eigen::VectorXd high_kp_vector;
  Eigen::VectorXd high_kd_vector;

  size_t dofNumber_with_floatingBase = 0; // Number of degrees of freedom in the robot
  size_t dofNumber = 0; // Number of degrees of freedom in the robot without floating base

  // For position control
  Eigen::VectorXd ddot_qp; // Desired acceleration in the QP solver
  Eigen::VectorXd ddot_qp_w_floatingBase; // Desired acceleration in the QP solver with floating base
  Eigen::VectorXd q_cmd; // The commended position send to the internal PD of the robot
  Eigen::VectorXd q_cmd_w_floatingBase; // The commended position send to the internal PD of the robot with floating base
  Eigen::VectorXd tau_cmd_after_pd; // The commended position after PD control

  // // For RL
  std::map<std::string, double> q_zero; // Reference position for the RL controller
  std::map<std::string, double> q_rl; // Target position for the RL controller: q_rl = q_zero + 0.75*a + 0.25*a_before
  std::map<std::string, double> a; // Action for the RL controller
  std::map<std::string, double> a_before; // Last action applied for the RL controller
  Eigen::VectorXd legPos, legVel, legAction; // Leg position, velocity and action in mc_rtc order

  std::vector<std::string> notControlledJoints; // Joints that are not controlled by the RL controller: arms + torso, in that case q_rl = q_zero
  std::vector<std::string> maniskillJointsOrder;
  std::vector<std::string> maniskillLegsJointsOrder; // Joint order in mc_rtc, used for reordering actions and observations
  std::vector<std::string> mcRtcJointsOrder;
  std::vector<int> legIndicesInManiskill; // Indices of the leg joints in the Maniskill order

  // Alice **************************************************************************
  std::unique_ptr<RLPolicyInterface> rlPolicy_;
  std::vector<std::string> legJoints_;      // 10 leg joints only
  
  std::vector<int> maniskillToMcRtcIdx_;    // Action reordering: Maniskill -> mc_rtc
  std::vector<int> mcRtcToManiskillIdx_;    // Observation reordering: mc_rtc -> Maniskill
  
  // Past action for observation (size 19)
  Eigen::VectorXd a_maniskillOrder;
  
  // Reference position and action blending for policy output
  Eigen::VectorXd q_zero_vector;               // Reference joint positions (19 joints in mc_rtc order)
  Eigen::VectorXd a_before_vector;             // Last actions applied (19 joints in mc_rtc order)
  Eigen::VectorXd a_vector;                    // Action in mc_rtc order
  
  // CSV data storage
  std::vector<Eigen::VectorXd> csvInputData_;  // Stores parsed CSV input data for each inference step
  std::atomic<int> inferenceCounter_;          // Thread-safe inference counter for CSV indexing
  
  // 40Hz inference timing
  std::chrono::steady_clock::time_point lastInferenceTime_;
  Eigen::VectorXd q_rl_vector;  // Hold target position between inference calls
  bool targetPositionValid_;               // Flag to check if we have a valid target
  static constexpr double INFERENCE_PERIOD_MS = 25.0;  // 40Hz = 25ms period
  
  // Eigen::VectorXd kp_;
  // Eigen::VectorXd kd_;
  
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

  Eigen::Vector3d baseAngVel; // Angular velocity of the base
  Eigen::Vector3d rpy; // Roll, Pitch, Yaw angles of the base

  // State-specific data
  size_t stepCount_ = 0;
  double startTime_ = 0.0;
  
  // Timing and statistics
  bool logTiming_ = false;
  size_t timingLogInterval_ = 1000;  // Log every N steps

  //QP 
  bool useQP = true;

  bool static_pos = false; // Static position control flag

  void logging();

  Eigen::VectorXd getCurrentObservation();
  void applyAction(const Eigen::VectorXd & action);
  
  // Threading
  void startInferenceThread();
  void stopInferenceThread();
  void inferenceThreadFunction();
  void updateObservationForInference();
  Eigen::VectorXd getLatestAction();
  
  // reordering methods
  Eigen::VectorXd reorderJointsToManiskill(const Eigen::VectorXd & obs);
  Eigen::VectorXd reorderJointsFromManiskill(const Eigen::VectorXd & action);
  
  // Eigen::VectorXd computeImpedanceTorques(const Eigen::VectorXd & desiredPos, 
  //                                        const Eigen::VectorXd & currentPos,
  //                                        const Eigen::VectorXd & currentVel);

  void supported_robots(std::vector<std::string> & out) const override 
  { 
    out = {"h1"}; 
  }

private:
  void initializeAllJoints();
  void loadCSVInputData();  // Parse and store all CSV input data during initialization
  // void initializeImpedanceGains();
}; 