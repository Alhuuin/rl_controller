#pragma once

#include <mc_control/fsm/Controller.h>
#include <mc_tasks/TorqueTask.h>
#include <mc_tasks/PostureTask.h>

#include "api.h"

#include "RLPolicyInterface.h"
#include "PolicySimulatorHandling.h"
#include "utils.h"

#include <memory>
#include <Eigen/Dense>

#include <chrono>
#include <vector>


#define TORQUE_TASK 0
#define FD_TASK 1
#define PURE_RL 2

struct RLController_DLLAPI RLController : public mc_control::fsm::Controller
{
  RLController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config);

  bool run() override;

  void reset(const mc_control::ControllerResetData & reset_data) override;

  void addLog();
  void addGui();
  void initializeRobot(const mc_rtc::Configuration & config);
  void initializeRLPolicy(const mc_rtc::Configuration & config);
  void initializeState();

  void updateRobotCmdAfterQP();
  void computeInversePD(); // Update q_cmd based on QP acceleration

  void tasksComputation(Eigen::VectorXd & currentTargetPosition);
  std::tuple<Eigen::VectorXd, Eigen::VectorXd> getPDGains();
  bool setPDGains(Eigen::VectorXd p_vec, Eigen::VectorXd d_vec);
  bool isHighGain(double tol = 1e-9);
  std::pair<sva::PTransformd, Eigen::Vector3d>  createContactAnchor(const mc_rbdyn::Robot & anchorRobot);

  void computeRLStateSimulated(); // Compute the state of the robot as if it was simulated with the RL policy

  // Task
  std::shared_ptr<mc_tasks::TorqueTask> torqueTask;

  std::map<std::string, std::vector<double>> torque_target; // Target torques for the torque task;
  bool useQP = true;
  bool isTorqueControl = false;

  // Robot specific data
  std::vector<std::string> jointNames;
  size_t dofNumber_with_floatingBase = 0;
  size_t dofNumber = 0;

  // Gains
  Eigen::VectorXd kp_vector;
  Eigen::VectorXd kd_vector;
  Eigen::VectorXd current_kp;
  Eigen::VectorXd current_kd;

  // Options
  bool compensateExternalForces = false;

  // Robot state 
  Eigen::VectorXd refAccel;
  Eigen::VectorXd tau_d;  // torque sends to a task
  Eigen::VectorXd currentPos;
  Eigen::VectorXd currentVel;
  Eigen::VectorXd currentTau;

  // For position control
  Eigen::VectorXd ddot_qp;
  Eigen::VectorXd ddot_qp_w_floatingBase;
  Eigen::VectorXd tau_cmd; // Torque computed from the QP acceleration
  Eigen::VectorXd tau_err; // Difference between tau_cmd and tau_rl
  double tau_err_norm = 0.0;
  Eigen::VectorXd qddot_err;
  double qddot_err_norm = 0.0;
  Eigen::VectorXd q_cmd;
  Eigen::VectorXd q_cmd_w_floatingBase;

  // For RL
  Eigen::VectorXd q_zero_vector;               // Reference joint positions
  Eigen::VectorXd a_before_vector;             // Last actions applied
  Eigen::VectorXd a_vector;                    // Action in mc_rtc order

  std::vector<std::string> notControlledJoints; // Joints that are not controlled by the RL controller: arms + torso, in that case q_rl = q_zero
  std::vector<std::string> mcRtcJointsOrder;
  
  Eigen::VectorXd a_simuOrder;
  std::vector<int> usedJoints_simuOrder; // Indices of the leg joints in the Maniskill order

  // RL policy 
  std::unique_ptr<RLPolicyInterface> rlPolicy_;
  std::unique_ptr<PolicySimulatorHandling> policySimulatorHandling_;
  utils utils_; // Utility functions for RL controller
  
  std::chrono::steady_clock::time_point lastInferenceTime_;
  Eigen::VectorXd q_rl;  // Hold target position between inference calls
  Eigen::VectorXd q_rl_simulatedMeasure;
  Eigen::VectorXd qdot_rl_simulatedMeasure;
  Eigen::VectorXd qddot_rl_simulatedMeasure;
  Eigen::VectorXd tau_rl; // Only use for logging

  // observation data - Policy specific
  Eigen::Vector3d baseAngVel; // Angular velocity of the base
  Eigen::Vector3d rpy; // Roll, Pitch, Yaw angles of the base
  Eigen::VectorXd legPos, legVel, legAction; // Leg position, velocity and action in mc_rtc order

  Eigen::Vector3d baseAngVel_prev; // Angular velocity of the base
  Eigen::Vector3d rpy_prev; // Roll, Pitch, Yaw angles of the base
  Eigen::VectorXd legPos_prev, legVel_prev, legAction_prev; // Leg position, velocity and action in mc_rtc order

  Eigen::Vector3d baseAngVel_prev_prev; // Angular velocity of the base
  Eigen::Vector3d rpy_prev_prev; // Roll, Pitch, Yaw angles of the base
  Eigen::VectorXd legPos_prev_prev, legVel_prev_prev, legAction_prev_prev; // Leg position, velocity and action in mc_rtc order

  Eigen::Vector3d velCmdRL_;                        // Command vector [vx, vy, yaw_rate]
  double phase_;                               // Current phase for periodic gait
  double phaseFreq_ = 1.2;                           // Phase frequency (1.2 Hz)
  std::chrono::steady_clock::time_point startPhase_; // Start time for phase calculation
    
  Eigen::VectorXd currentObservation_;   // Protected by observationMutex_
  Eigen::VectorXd currentAction_;        // Protected by actionMutex_
  Eigen::VectorXd latestAction_;         // current action being used by control loop
  
  bool useAsyncInference_;

  // Timing and statistics
  bool logTiming_ = false;
  size_t timingLogInterval_ = 1000;  // Log every N steps

  // Log constraints
  Eigen::Vector3d leftAnklePos; 
  Eigen::Vector3d rightAnklePos;
  double ankleDistanceNorm;

  double velPercent = 0.9;
  double dsPercent = 0.01;
  double diPercent = 0.2;

  Eigen::VectorXd jointLimitsPos_upper;
  Eigen::VectorXd jointLimitsPos_lower;
  Eigen::VectorXd jointLimitsHardPos_upper;
  Eigen::VectorXd jointLimitsHardPos_lower;
  Eigen::VectorXd jointLimitsVel_upper;
  Eigen::VectorXd jointLimitsVel_lower;
  Eigen::VectorXd jointLimitsHardVel_upper;
  Eigen::VectorXd jointLimitsHardVel_lower;
  Eigen::VectorXd jointLimitsHardTau_upper;
  Eigen::VectorXd jointLimitsHardTau_lower;

  Eigen::VectorXd limitBreached_q_soft_upper;
  Eigen::VectorXd limitBreached_q_soft_lower;
  Eigen::VectorXd limitBreached_q_hard_upper;
  Eigen::VectorXd limitBreached_q_hard_lower;
  Eigen::VectorXd limitBreached_qDot_soft_upper;
  Eigen::VectorXd limitBreached_qDot_soft_lower;
  Eigen::VectorXd limitBreached_qDot_hard_upper;
  Eigen::VectorXd limitBreached_qDot_hard_lower;
  Eigen::VectorXd limitBreached_tau_upper;
  Eigen::VectorXd limitBreached_tau_lower;

  // State after QP without any modification
  std::vector<std::vector<double>> qOut;
  std::vector<std::vector<double>> alphaOut;
  std::vector<std::vector<double>> tauOut;

  // Floating base
  Eigen::VectorXd floatingBase_qOut;
  Eigen::VectorXd floatingBase_alphaOut;
  Eigen::VectorXd floatingBase_tauOut;

  Eigen::VectorXd floatingBase_qOutPD;
  Eigen::VectorXd floatingBase_alphaOutPD;
  Eigen::VectorXd floatingBase_tauOutPD;

  Eigen::VectorXd floatingBase_qIn;
  Eigen::VectorXd floatingBase_alphaIn;

  double counter = 0.0; // Time counter in seconds
};