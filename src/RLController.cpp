#include "RLController.h"

#include <RBDyn/MultiBodyConfig.h>
#include <cstddef>
#include <mc_rtc/gui/ArrayInput.h>
#include <mc_rtc/gui/NumberSlider.h>
#include <mc_rtc/gui/Transform.h>

#include <mc_joystick_plugin/joystick_inputs.h>

#include <fcntl.h>
#include <numeric>
#include <termios.h>


RLController::RLController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(rm, dt, config, Backend::TVM)
{
  currentPolicyIndex = size_t(config("default_policy_index", 0));
  initializeRobot(config);
  initializeRLPolicy(config);

  addGui(config);
  addLog();
  mc_rtc::log::success("RLController init");
}

bool RLController::run()
{
  // Test joystick inputs
  if(datastore().has("Joystick::connected") && datastore().get<bool>("Joystick::connected"))
    RLuseJoyStickInputs();
  else
    RLuseKeyboardInputs();
  
  bool run = manageModeSwitching();
  if(byPassQPControl()) // Run RL without taking the QP into account
  {
    return true;
  }
  return run; // Return false if QP fails
}

void RLController::reset(const mc_control::ControllerResetData & reset_data)
{
  mc_control::fsm::Controller::reset(reset_data);
  mc_rtc::log::success("RLController reset completed");
}

void RLController::initializeRobot(const mc_rtc::Configuration & config)
{
  useQP_ = config("policies")[currentPolicyIndex]("use_QP", true);
  isTorqueControl_ = config("policies")[currentPolicyIndex]("is_torque_control", false);
  if(isTorqueControl_)
  {
    mc_rtc::log::info("Using Torque Control mode");
    datastore().make<std::string>("ControlMode", "Torque");
  }
  else
  {
    mc_rtc::log::info("Using Position Control mode");
    datastore().make<std::string>("ControlMode", "Position");
  }

  // get the joints order (urdf) depending on the robot used
  robotName_ = robot().name();
  dofNumber = robot().mb().nrDof() - 6; // Remove the floating base part (6 DoF)

  q_rl = Eigen::VectorXd::Zero(dofNumber);
  q_zero_vector = Eigen::VectorXd::Zero(dofNumber);
  kp_ = Eigen::VectorXd::Zero(dofNumber);
  kd_ = Eigen::VectorXd::Zero(dofNumber);
  kpBase_ = Eigen::VectorXd::Zero(dofNumber);
  kdBase_ = Eigen::VectorXd::Zero(dofNumber);
  
  // Get the gains from the configuration or set default values
  pdGainsRatio_ = config("policies")[currentPolicyIndex]("pd_gains_ratio", 1.0);
  std::map<std::string, double> kp_map = config("policies")[currentPolicyIndex]("kp");
  std::map<std::string, double> kd_map = config("policies")[currentPolicyIndex]("kd");
  // Get the default posture target from the robot's posture task
  std::shared_ptr<mc_tasks::PostureTask> FSMPostureTask = getPostureTask(robot().name());
  auto posture = FSMPostureTask->posture();
  int i = 0;
  std::vector<std::string> joint_names;
  joint_names.reserve(robot().mb().joints().size());
  
  for (const auto &j : robot().mb().joints()) {
      const std::string &joint_name = j.name();
      if(j.type() == rbd::Joint::Type::Rev)
      {
        jointNames_.emplace_back(joint_name);  
        mc_rtc::log::info("[RLController] Found joint: {}", joint_name);
        if (const auto &t = posture[robot().jointIndexByName(joint_name)]; !t.empty()) {
            kpBase_[i] = kp_map.at(joint_name);
            kdBase_[i] = kd_map.at(joint_name);
            q_rl[i] = t[0];
            q_zero_vector[i] = t[0];
            i++;
        }
      }
  }

  

  solver().removeTask(FSMPostureTask);
  datastore().make_call("anchorFrameFunction", 
    [this](const mc_rbdyn::Robot & real_robot) {return createContactAnchor(real_robot);});
  
  //Initialize Constraints
  selfCollisionConstraint->setCollisionsDampers(solver(), {1.2, 20.0});
  solver().removeConstraintSet(dynamicsConstraint);
  dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(
    new mc_solver::DynamicsConstraint(robots(), 0, {diPercent_, dsPercent_, 0.0, 1.2, 100.0}, velPercent_, true));
  solver().addConstraintSet(dynamicsConstraint);

  // Initialize Task
  torqueJointTask = std::make_shared<mc_tasks::TorqueJointTask>(
      solver(), robot().robotIndex(), 100.0, 1);

  kp_ = pdGainsRatio_ * kpBase_;
  kd_ = pdGainsRatio_ * kdBase_;
  torqueJointTask->setStiffness(kp_);
  torqueJointTask->setDamping(kd_);
  torqueJointTask->setPosTarget(q_rl);
}

void RLController::initializeRLPolicy(const mc_rtc::Configuration & config)
{
  // load policy specific configuration
  policyPaths_ = config("policy_path", std::vector<std::string>{"walking_better_h1.onnx"});
  configRL(config);
  auto & real_robot = realRobot(robots()[0].name());
  
  std::string baseName = "pelvis";
  auto & imu = robot().bodySensor("Accelerometer");
  baseAngVel = imu.angularVelocity();

  Eigen::Matrix3d baseRot = real_robot.bodyPosW(baseName).rotation();
  rpy = mc_rbdyn::rpyFromMat(baseRot);
    
  mc_rtc::log::info("[RLController] Posture target initialized with {} joints", dofNumber); 

  // Initialize reference position and last actions for action blending
  int vect_size = int(usedJoints_mcRtcOrder.size());
  a_vector = Eigen::VectorXd::Zero(dofNumber);
  legPos = Eigen::VectorXd::Zero(vect_size);
  legVel = Eigen::VectorXd::Zero(vect_size);
  legAction = Eigen::VectorXd::Zero(vect_size);

  baseAngVel_prev = Eigen::Vector3d::Zero();
  rpy_prev = Eigen::Vector3d::Zero();
  legPos_prev = Eigen::VectorXd::Zero(vect_size);
  legVel_prev = Eigen::VectorXd::Zero(vect_size);
  legAction_prev = Eigen::VectorXd::Zero(vect_size);

  baseAngVel_prev_prev = Eigen::Vector3d::Zero();
  rpy_prev_prev = Eigen::Vector3d::Zero();
  legPos_prev_prev = Eigen::VectorXd::Zero(vect_size);
  legVel_prev_prev = Eigen::VectorXd::Zero(vect_size);
  legAction_prev_prev = Eigen::VectorXd::Zero(vect_size);

  a_simuOrder = Eigen::VectorXd::Zero(dofNumber);
  
  currentAction = Eigen::VectorXd::Zero(dofNumber);
  
  // Initialize new observation components
  velCmdRL = Eigen::Vector3d::Zero();  // Default command (x, y, yaw)
  phase = 0.0;  // Phase for periodic gait
}

void RLController::switchPolicy(int policyIndex, const mc_rtc::Configuration & config)
{
  if(policyIndex < 0 || policyIndex >= static_cast<int>(policyPaths_.size())) {
    mc_rtc::log::error("Invalid policy index: {}", policyIndex);
    return;
  }
  
  mc_rtc::log::info("Switching from policy [{}] to policy [{}]", currentPolicyIndex, policyIndex);
  currentPolicyIndex = size_t(policyIndex);
  
  // Update policy-specific boolean flags
  useQP_ = config("policies")[currentPolicyIndex]("use_QP", true);
  isTorqueControl_ = config("policies")[currentPolicyIndex]("is_torque_control", false);
  if(isTorqueControl_) datastore().get<std::string>("ControlMode") = "Torque";
  else datastore().get<std::string>("ControlMode") = "Position";

  configRL(config);

  // Update PD gains
  pdGainsRatio_ = config("policies")[currentPolicyIndex]("pd_gains_ratio", 1.0);
  std::map<std::string, double> kp_map = config("policies")[currentPolicyIndex]("kp");
  std::map<std::string, double> kd_map = config("policies")[currentPolicyIndex]("kd");

  for(int i = 0; i < dofNumber; ++i) {
    const auto & jName = robot().mb().joint(static_cast<int>(i + 1)).name();  // +1 to skip Root
    if(kp_map.count(jName)) {
      kpBase_(i) = kp_map[jName];
    }
    if(kd_map.count(jName)) {
      kdBase_(i) = kd_map[jName];
    }
  }
  // Update PD gains
  kp_ = pdGainsRatio_ * kpBase_;
  kd_ = pdGainsRatio_ * kdBase_;
  torqueJointTask->setStiffness(kp_);
  torqueJointTask->setDamping(kd_);
}

bool RLController::byPassQPControl()
{
  if(useQP_) return false; // QP is not bypassed, do nothing
  if(!isTorqueControl_)
  {
    mc_rtc::log::warning("[RLController] QP can't be bypassed in position control mode.");
    mc_rtc::log::warning("[RLController] QP is enforced.");
    return false;
  }

  robot().forwardKinematics();
  robot().forwardVelocity();
  robot().forwardAcceleration();

  auto tau = robot().mbc().jointTorque;
  auto q_map = robot().encoderValues();
  auto q_dot_map = robot().encoderVelocities();

  Eigen::VectorXd q = Eigen::VectorXd::Map(q_map.data(), int(q_map.size()));
  Eigen::VectorXd q_dot = Eigen::VectorXd::Map(q_dot_map.data(), int(q_dot_map.size()));
  Eigen::VectorXd tau_rl = (kp_).cwiseProduct(q_rl - q) - (kd_).cwiseProduct(q_dot);
  
  int i = 0;
  for (const auto &joint_name : jointNames_)
  {
    tau[robot().jointIndexByName(joint_name)][0] = tau_rl[i];
    i++;
  }

  // Update joint torques for torque control
  robot().mbc().jointTorque = tau;
  return true;
}

void RLController::addLog()
{
  // Robot State variables
  logger().addLogEntry("RLController_kp_base", [this]() { return kpBase_; });
  logger().addLogEntry("RLController_kd_base", [this]() { return kdBase_; });
  logger().addLogEntry("RLController_kp_current", [this]() { return kp_; });
  logger().addLogEntry("RLController_kd_current", [this]() { return kd_; });
  logger().addLogEntry("RLController_pd_gains_ratio", [this]() { return pdGainsRatio_; });

  // RL variables
  logger().addLogEntry("RLController_RL_q", [this]() { return q_rl; });
  logger().addLogEntry("RLController_RL_pastAction", [this]() { return a_simuOrder; });
  logger().addLogEntry("RLController_RL_qZero", [this]() { return q_zero_vector; });
  logger().addLogEntry("RLController_RL_currentObservation", [this]() { return currentObservation; });
  logger().addLogEntry("RLController_RL_a_vector", [this]() { return a_vector; });
  logger().addLogEntry("RLController_RL_a_simulationOrder", [this]() { return a_simuOrder; });
  logger().addLogEntry("RLController_RL_currentAction", [this]() { return currentAction; });
  logger().addLogEntry("RLController_RL_baseAngVel", [this]() { return baseAngVel; });
  logger().addLogEntry("RLController_RL_rpy", [this]() { return rpy; });
  logger().addLogEntry("RLController_RL_legPos", [this]() { return legPos; });
  logger().addLogEntry("RLController_RL_legVel", [this]() { return legVel; });
  logger().addLogEntry("RLController_RL_legAction", [this]() { return legAction; });
  logger().addLogEntry("RLController_RL_phase", [this]() { return phase; });
  
  // Controller state variables
  logger().addLogEntry("RLController_useQP", [this]() { return useQP_; });
  logger().addLogEntry("RLController_isTorqueControl", [this]() { return isTorqueControl_; });

  // Log current policy (combined index and path)
  logger().addLogEntry("RLController_currentPolicy", [this]() { 
    return std::to_string(currentPolicyIndex) + ": " + policyPaths_[currentPolicyIndex]; 
  });
}

void RLController::addGui(const mc_rtc::Configuration & config)
{
  gui()->addElement({"RLController", "Policy"},
  mc_rtc::gui::ArrayInput("Velocity Command RL", {"X", "Y", "Yaw"}, velCmdRL),
    mc_rtc::gui::NumberInput("Max X Y Vel via Joystick", maxVelCmd_),
    mc_rtc::gui::NumberInput("Max yaw via Joystick", maxYawCmd_),
    mc_rtc::gui::Label("Current policy", [this]() -> const std::string & 
    { 
      return policyPaths_[currentPolicyIndex]; 
    }),
    mc_rtc::gui::ComboInput(
      "Select policy",
      policyPaths_,
      [this]() -> const std::string & 
      { 
        return policyPaths_[currentPolicyIndex]; 
      },
      [this, config](const std::string & selected) 
      {  // Capture config by VALUE (makes a safe copy)
        // Find the index of the selected policy
        auto it = std::find(policyPaths_.begin(), policyPaths_.end(), selected);
        if(it != policyPaths_.end()) 
        {
          int newIndex = static_cast<int>(std::distance(policyPaths_.begin(), it));
          mc_rtc::log::info("User requested policy switch to [{}]: {}", newIndex, selected);
          // Switch to new policy without reinitializing robot
          switchPolicy(newIndex, config);
        }
      }),
    mc_rtc::gui::Button("Reload current policy", [this, config]() 
    {
      mc_rtc::log::info("User requested to reload current policy [{}]", currentPolicyIndex);
      switchPolicy(int(currentPolicyIndex), config);
    })
  );

  // Add PD gains ratio slider
  gui()->addElement({"RLController", "PD Gains"},
    mc_rtc::gui::NumberSlider(
      "PD Gains Ratio", [this]() { return pdGainsRatio_; },
      [this](double v) { 
        pdGainsRatio_ = v;
        kp_ = pdGainsRatio_ * kpBase_;
        kd_ = pdGainsRatio_ * kdBase_;
        torqueJointTask->setStiffness(kp_);
        torqueJointTask->setDamping(kd_);
      }, 0.0, 2.0),
    mc_rtc::gui::Label("Current kp", kp_),
    mc_rtc::gui::Label("Current kd", kd_)
  );
  
  gui()->addElement({"RLController", "Options"},
    mc_rtc::gui::Transform("Anchor Frame",contactAnchorTf_)
  );

  gui()->addElement({"ControlMode"}, 
    mc_rtc::gui::Button("Switch Control Mode", [this]()
      {
        controlModeChanged_ = true;
        isTorqueControl_ = !isTorqueControl_;
      }),
      mc_rtc::gui::Label("Current Control Mode", [this]()
        {
          return isTorqueControl_ ? "Torque Control" : "Position Control";
        }),
      mc_rtc::gui::Button("Toggle QP Control", [this]()
        {
          useQP_ = !useQP_;
        }),
      mc_rtc::gui::Label("QP Control", [this]()
      {
        return useQP_ ? "Enforced" : "Bypassed";
      })
    );
}

void RLController::configRL(const mc_rtc::Configuration & config)
{
  mc_rtc::log::info("Loading RL policy [{}]: {}", currentPolicyIndex, policyPaths_[currentPolicyIndex]);
  try {
    rlPolicy = std::make_unique<RLPolicyInterface>(policyPaths_[currentPolicyIndex]);
    if(rlPolicy) {
      mc_rtc::log::success("RL policy loaded successfully");
      // Initialize observation vector with the correct size from the loaded policy
      currentObservation = Eigen::VectorXd::Zero(rlPolicy->getObservationSize());
      mc_rtc::log::info("Initialized observation vector with size: {}", rlPolicy->getObservationSize());
    } else {
      mc_rtc::log::error_and_throw("RL policy creation failed - policy is null");
    }
  } catch(const std::exception& e) {
    mc_rtc::log::error_and_throw("Failed to load RL policy: {}", e.what());
  }

  std::string simulator = config("policies")[currentPolicyIndex]("simulator", std::string(""));
  if (simulator.empty())
  {
    mc_rtc::log::warning("Simulator not set, using default handling");
    policySimulatorHandling = std::make_unique<PolicySimulatorHandling>();
  }
  else
  {
    mc_rtc::log::info("Using {} handling", simulator);
    policySimulatorHandling = std::make_unique<PolicySimulatorHandling>(simulator, robotName_);
  }

  // get list of used joints from config
  usedJoints_mcRtcOrder = config("policies")[currentPolicyIndex]("used_joints_index", std::vector<int>{});
  if(!usedJoints_mcRtcOrder.empty())
  {
    std::string jointsStr = "[";
    for(size_t i = 0; i < usedJoints_mcRtcOrder.size(); ++i) {
      if(i > 0) jointsStr += ", ";
      jointsStr += std::to_string(usedJoints_mcRtcOrder[i]);
    }
    jointsStr += "]";
    mc_rtc::log::info("Using custom used joints: {}", jointsStr);
    usedJoints_simuOrder = policySimulatorHandling->getSimulatorIndices(usedJoints_mcRtcOrder);
    std::sort(usedJoints_simuOrder.begin(), usedJoints_simuOrder.end());
    jointsStr = "[";
    for(size_t i = 0; i < usedJoints_simuOrder.size(); ++i) {
      if(i > 0) jointsStr += ", ";
      jointsStr += std::to_string(usedJoints_simuOrder[i]);
    }
    jointsStr += "]";
    mc_rtc::log::info("Using custom used joints: {}", jointsStr);

  }
  else {
    mc_rtc::log::info("No custom used joints specified, using default all joints");
    usedJoints_simuOrder = std::vector<int>(size_t(dofNumber));
    std::iota(usedJoints_simuOrder.begin(), usedJoints_simuOrder.end(), 0);
  }
  maxVelCmd_ = config("policies")[currentPolicyIndex]("speed_multiplier_joystick", 0.6);
  maxYawCmd_ = config("policies")[currentPolicyIndex]("max_yaw_joystick", 0.7);

  actionScale = config("policies")[currentPolicyIndex]("action_scale", 1.0);
  policyPeriodMs = config("policies")[currentPolicyIndex]("policy_period_ms", 20.0);
}

std::pair<sva::PTransformd, Eigen::Vector3d> RLController::createContactAnchor(const mc_rbdyn::Robot & anchorRobot)
{
  sva::PTransformd X_foot_r = anchorRobot.bodyPosW("right_ankle_link");
  sva::PTransformd X_foot_l = anchorRobot.bodyPosW("left_ankle_link");

  sva::MotionVecd v_foot_r = anchorRobot.bodyVelW("right_ankle_link");
  sva::MotionVecd v_foot_l = anchorRobot.bodyVelW("left_ankle_link");

  auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
  int right_knee_index = int(robot().jointIndexByName("right_knee_joint")) + 5;
  int left_knee_index = int(robot().jointIndexByName("left_knee_joint")) + 5;
  double tau_ext_knee_r =  abs(extTorqueSensor.torques()[right_knee_index]);
  double tau_ext_knee_l =  abs(extTorqueSensor.torques()[left_knee_index]);
  double leftFootRatio = tau_ext_knee_l/(tau_ext_knee_r+tau_ext_knee_l);
  if(tau_ext_knee_r + tau_ext_knee_l < 0.02)
  {
    leftFootRatio = 0.5;
  }
         
  Eigen::VectorXd w_r = X_foot_r.translation();
  Eigen::VectorXd w_l = X_foot_l.translation();
  Eigen::VectorXd contact_anchor = (w_r * (1 - leftFootRatio) + w_l * leftFootRatio)  ;
  Eigen::VectorXd anchor_vel = (v_foot_r.linear() * (1 - leftFootRatio) + v_foot_l.linear() * leftFootRatio);
  contactAnchorTf_ = sva::PTransformd(Eigen::Matrix3d::Identity(), contact_anchor); 

  return {contactAnchorTf_, anchor_vel};
}

void RLController::RLuseJoyStickInputs()
{
  // Get joystick functions
  auto & stickFunc = datastore().get<std::function<Eigen::Vector2d(joystickAnalogicInputs)>>("Joystick::Stick");
  
  // Read sticks values
  leftStick_ = stickFunc(joystickAnalogicInputs::L_STICK);
  // Apply dead zone
  double vel_x = 0.0;
  if(std::abs(leftStick_(0) - 0.5) > joystickDeadZone_)
  {
    vel_x = (leftStick_(0) - 0.5) * 2.0 * maxVelCmd_;
  }
  double vel_y = 0.0;
  if(std::abs(leftStick_(1) - 0.5) > joystickDeadZone_)
  {
    vel_y = (leftStick_(1) - 0.5) * 2.0 * maxVelCmd_;
  }
  velCmdRL(0) = vel_x;
  velCmdRL(1) = vel_y;

  rightStick_ = stickFunc(joystickAnalogicInputs::R_STICK);
  double yaw_cmd = 0.0;
  if(std::abs(rightStick_(1) - 0.5) > joystickDeadZone_)
  {
    yaw_cmd = (rightStick_(1) - 0.5) * 2.0 * maxYawCmd_;
  }
  velCmdRL(2) = yaw_cmd;
  
  // Read D-pad buttons
  directionButtons_ = {
    datastore().get<bool>("Joystick::UpPad"),
    datastore().get<bool>("Joystick::DownPad"),
    datastore().get<bool>("Joystick::LeftPad"),
    datastore().get<bool>("Joystick::RightPad")
  };

  for (size_t i = 0; i < directionButtons_.size(); ++i)
  {
    if(directionButtons_[i])
    {
      switch(i)
      {
        case 0: // Up
          velCmdRL(0) += 1.0 * maxVelCmd_;
          break;
        case 1: // Down
          velCmdRL(0) -= 1.0 * maxVelCmd_;
          break;
        case 2: // Left
          velCmdRL(1) += 1.0 * maxVelCmd_;
          break;
        case 3: // Right
          velCmdRL(1) -= 1.0 * maxVelCmd_;
          break;
        default:
          break;
      }
    }
  }
}

void RLController::RLuseKeyboardInputs()
{
  struct Ctx { bool ready = false; termios old{}; bool seen[4] = {}; 
    std::chrono::steady_clock::time_point ts[4]; std::array<char, 64> buf{}; size_t sz = 0; };
  static Ctx k;

  if(!k.ready) {
    if(::isatty(STDIN_FILENO) != 1) return;
    k.ready = true;
    ::tcgetattr(STDIN_FILENO, &k.old);
    termios raw = k.old; raw.c_lflag &= static_cast<tcflag_t>(~static_cast<tcflag_t>(ICANON | ECHO)); 
    raw.c_cc[VMIN] = raw.c_cc[VTIME] = 0;
    ::tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    ::fcntl(STDIN_FILENO, F_SETFL, ::fcntl(STDIN_FILENO, F_GETFL, 0) | O_NONBLOCK);
  }

  char tmp[32]; ssize_t n = ::read(STDIN_FILENO, tmp, sizeof(tmp));
  if(n > 0 && k.sz + size_t(n) < 64) std::copy(tmp, tmp + n, k.buf.begin() + k.sz), k.sz += size_t(n);
  
  auto now = std::chrono::steady_clock::now();
  for(size_t i = 0; i + 2 < k.sz; ++i)
    if(k.buf[i] == 27 && k.buf[i + 1] == '[') {
      int idx = k.buf[i + 2] == 'A' ? 0 : k.buf[i + 2] == 'B' ? 1 : k.buf[i + 2] == 'D' ? 2 : k.buf[i + 2] == 'C' ? 3 : -1;
      if(idx >= 0) k.seen[idx] = true, k.ts[idx] = now;
      i += 2;
    }
  std::copy(k.buf.begin() + (k.sz > 2 ? k.sz - 2 : 0), k.buf.begin() + k.sz, k.buf.begin());
  k.sz = k.sz > 2 ? 2 : 0;

  const auto active = [&](int i) { return k.seen[i] && 
    std::chrono::duration_cast<std::chrono::milliseconds>(now - k.ts[i]).count() < 500; };
  velCmdRL.setZero();
  velCmdRL(0) = (active(0) ? maxVelCmd_ : 0.0) - (active(1) ? maxVelCmd_ : 0.0);
  velCmdRL(1) = (active(2) ? maxVelCmd_ : 0.0) - (active(3) ? maxVelCmd_ : 0.0);
}

bool RLController::manageModeSwitching()
{
  if(controlModeChanged_)
  {
    if(isTorqueControl_)
    {
      mc_rtc::log::info("Switching to Torque Control");
      datastore().assign<std::string>("ControlMode", "Torque");
    }
    else
    {
      mc_rtc::log::info("Switching to Position Control");
      datastore().assign<std::string>("ControlMode", "Position");
    }
    controlModeChanged_ = false;
  }

  if(isTorqueControl_)
  {
    return mc_control::fsm::Controller::run(
          mc_solver::FeedbackType::ClosedLoopIntegrateReal);
  }
  else 
  {
    return mc_control::fsm::Controller::run();
  }
}