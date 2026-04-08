#pragma once

#include <mc_control/fsm/Controller.h>
#include <mc_tasks/TorqueJointTask.h>

#include "api.h"

#include "RLPolicyInterface.h"
#include "PolicySimulatorHandling.h"
#include "utils.h"

struct RLController_DLLAPI RLController : public mc_control::fsm::Controller
{
  RLController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config);

  bool run() override;
  void reset(const mc_control::ControllerResetData & reset_data) override;

  // Task
  std::shared_ptr<mc_tasks::TorqueJointTask> torqueJointTask;
  
  int dofNumber = 0;

  // Public RL related variables
  Eigen::VectorXd q_rl;

  double actionScale;
  double policyPeriodMs;
  Eigen::VectorXd q_zero_vector;               // Reference joint positions
  Eigen::VectorXd a_vector;                    // Action in mc_rtc order

  std::vector<int> usedJoints_mcRtcOrder; // Indices of the leg joints in the mc_rtc order
  std::vector<int> usedJoints_simuOrder; // Indices of the leg joints in the Simulator order
  Eigen::VectorXd a_simuOrder;

  size_t currentPolicyIndex = 0;
  std::unique_ptr<RLPolicyInterface> rlPolicy;
  std::unique_ptr<PolicySimulatorHandling> policySimulatorHandling;
  utils utilsClass; // Utility functions for RL controller

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

  Eigen::Vector3d velCmdRL;                        // Command vector [vx, vy, yaw_rate]
  double phase;                               // Current phase for periodic gait
  double phaseFreq = 1.2;                           // Phase frequency (1.2 Hz)
    
  Eigen::VectorXd currentObservation;
  Eigen::VectorXd currentAction;

  private:
    void addLog();
    void addGui(const mc_rtc::Configuration & config);

    void initializeRobot(const mc_rtc::Configuration & config);
    void configRL(const mc_rtc::Configuration & config);
    void initializeRLPolicy(const mc_rtc::Configuration & config);
    void switchPolicy(int policyIndex, const mc_rtc::Configuration & config);  // Switch to a different policy at runtime

    bool manageModeSwitching(); // Handle switching between Torque and Position control modes
    bool byPassQPControl(); // Directly use RL output without QP modifications
    
    std::pair<sva::PTransformd, Eigen::Vector3d> createContactAnchor(const mc_rbdyn::Robot & anchorRobot);

    void RLuseJoyStickInputs();
    void RLuseKeyboardInputs();

    std::string robotName_;
    std::vector<std::string> jointNames_;

    // Mode switching
    bool useQP_ = true;
    bool isTorqueControl_ = false;
    bool controlModeChanged_ = false;

    // Constraint configuration
    double velPercent_ = 0.95;
    double dsPercent_ = 0.01;
    double diPercent_ = 0.1;

    // Gains
    double pdGainsRatio_ = 1.0;
    Eigen::VectorXd kp_;  // Gains set to the robot/simulator = pd_gains_ratio * kp_base
    Eigen::VectorXd kd_;  // Gains set to the robot/simulator = pd_gains_ratio * kd_base
    Eigen::VectorXd kpBase_; // Base RL PD gains from config
    Eigen::VectorXd kdBase_; // Base RL PD gains from config

    // RL
    std::vector<std::string> policyPaths_;

    // Joystick input handling
    std::vector<bool> directionButtons_ = std::vector<bool>(4, false); // Up, Down, Left, Right
    double joystickDeadZone_ = 0.02; // Dead zone for joystick inputs
    Eigen::Vector2d leftStick_ = Eigen::Vector2d(0.5, 0.5); // x (UP), y (LEFT)
    Eigen::Vector2d rightStick_ = Eigen::Vector2d(0.5, 0.5); // x (UP), y (LEFT)
    double maxVelCmd_;
    double maxYawCmd_;

    // Anchor from for tilt estimation
    sva::PTransformd contactAnchorTf_;
};
