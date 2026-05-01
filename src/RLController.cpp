#include "RLController.h"
#include <Eigen/src/Core/VectorBlock.h>
#include <RBDyn/MultiBodyConfig.h>
#include <SpaceVecAlg/SpaceVecAlg>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <mc_rtc/Configuration.h>
#include <mc_rtc/gui/ArrayInput.h>
#include <mc_rtc/gui/NumberSlider.h>
#include <mc_rtc/gui/Transform.h>
#include <mc_rtc/logging.h>
#include <mc_rbdyn/configuration_io.h>
#include <mc_joystick_plugin/joystick_inputs.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <numeric>
#include <termios.h>
#include <unistd.h>
#include <utility>
#include <vector>

RLController::RLController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(rm, dt, config, Backend::TVM)
{
  loadConfig(config);

  //Initialize Constraints
  selfCollisionConstraint->setCollisionsDampers(solver(), {1.2, 200.0});
  solver().removeConstraintSet(dynamicsConstraint);
  dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(
    new mc_solver::DynamicsConstraint(robots(), 0, {diPercent, dsPercent, 0.0, 1.2, 200.0}, velPercent, true));
  solver().addConstraintSet(dynamicsConstraint);

  // Initialize Task
  torqueTask = std::make_shared<mc_tasks::TorqueTask>(solver(), robot().robotIndex());

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
  
  counter += timeStep;
  if (useResidual)
  {
    if(robotName == "H1")
    {
      leftAnklePos = robot().mbc().bodyPosW[robot().bodyIndexByName("left_ankle_link")].translation();
      rightAnklePos = robot().mbc().bodyPosW[robot().bodyIndexByName("right_ankle_link")].translation();
      ankleDistanceNorm = (leftAnklePos - rightAnklePos).norm();
    }
  }
  auto & real_robot = realRobot(robots()[0].name());

  auto qIn = real_robot.mbc().q;
  auto alphaIn = real_robot.mbc().alpha;
  floatingBase_qIn = rbd::paramToVector(robot().mb(), qIn);
  floatingBase_alphaIn = rbd::dofToVector(robot().mb(), alphaIn);
  Eigen::MatrixXd Kp_inv = (pd_gains_ratio * kp_vector).cwiseInverse().asDiagonal();
  if (useResidual)
  {
    if (robotName == "H1")
    {
      auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
      auto tau_ext = extTorqueSensor.torques();
    }
  }
  bool run = mc_control::fsm::Controller::run(mc_solver::FeedbackType::ClosedLoopIntegrateReal);
  robot().forwardKinematics();
  robot().forwardVelocity();
  robot().forwardAcceleration();
  if(!useQP) // Run RL without taking the QP into account
  {
    q_cmd = q_rl; // Use the RL position as the commanded position
    tau_cmd = (pd_gains_ratio * kp_vector).cwiseProduct(q_rl - currentPos) - (pd_gains_ratio * kd_vector).cwiseProduct(currentVel);
    computeRLStateSimulated();
    updateRobotCmdAfterQP();
    return true;
  }
  // Use QP
  computeInversePD();
  updateRobotCmdAfterQP();
  computeRLStateSimulated();
  return run; // Return false if QP fails
}

void RLController::reset(const mc_control::ControllerResetData & reset_data)
{
  mc_control::fsm::Controller::reset(reset_data);
  mc_rtc::log::success("RLController reset completed");
}

void RLController::RLuseJoyStickInputs()
{
  // Get joystick functions
  auto & stickFunc = datastore().get<std::function<Eigen::Vector2d(joystickAnalogicInputs)>>("Joystick::Stick");
  
  // Read sticks values
  leftStick = stickFunc(joystickAnalogicInputs::L_STICK);
  // Apply dead zone
  double vel_x = 0.0;
  if(std::abs(leftStick(0) - 0.5) > joystickDeadZone)
  {
    vel_x = (leftStick(0) - 0.5) * 2.0 * maxVelCmd;
  }
  double vel_y = 0.0;
  if(std::abs(leftStick(1) - 0.5) > joystickDeadZone)
  {
    vel_y = (leftStick(1) - 0.5) * 2.0 * maxVelCmd;
  }
  velCmdRL_(0) = vel_x;
  velCmdRL_(1) = vel_y;

  rightStick = stickFunc(joystickAnalogicInputs::R_STICK);
  double yaw_cmd = 0.0;
  if(std::abs(rightStick(1) - 0.5) > joystickDeadZone)
  {
    yaw_cmd = (rightStick(1) - 0.5) * 2.0 * maxYawCmd;
  }
  velCmdRL_(2) = yaw_cmd;
  
  // Read D-pad buttons
  DirectionButtons = {
    datastore().get<bool>("Joystick::UpPad"),
    datastore().get<bool>("Joystick::DownPad"),
    datastore().get<bool>("Joystick::LeftPad"),
    datastore().get<bool>("Joystick::RightPad")
  };

  for (size_t i = 0; i < DirectionButtons.size(); ++i)
  {
    if(DirectionButtons[i])
    {
      switch(i)
      {
        case 0: // Up
          velCmdRL_(0) += 1.0 * maxVelCmd;
          break;
        case 1: // Down
          velCmdRL_(0) -= 1.0 * maxVelCmd;
          break;
        case 2: // Left
          velCmdRL_(1) += 1.0 * maxVelCmd;
          break;
        case 3: // Right
          velCmdRL_(1) -= 1.0 * maxVelCmd;
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
    termios raw = k.old; raw.c_lflag &= ~(ICANON | ECHO); raw.c_cc[VMIN] = raw.c_cc[VTIME] = 0;
    ::tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    ::fcntl(STDIN_FILENO, F_SETFL, ::fcntl(STDIN_FILENO, F_GETFL, 0) | O_NONBLOCK);
  }

  char tmp[32]; ssize_t n = ::read(STDIN_FILENO, tmp, sizeof(tmp));
  if(n > 0 && k.sz + n < 64) std::copy(tmp, tmp + n, k.buf.begin() + k.sz), k.sz += n;
  
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
  velCmdRL_.setZero();
  velCmdRL_(0) = (active(0) ? maxVelCmd : 0.0) - (active(1) ? maxVelCmd : 0.0);
  velCmdRL_(1) = (active(2) ? maxVelCmd : 0.0) - (active(3) ? maxVelCmd : 0.0);
}


void RLController::loadConfig(const mc_rtc::Configuration & config)
{ 
  currentPolicyIndex = config("default_policy_index", 0);
  robotName = config("policies")[currentPolicyIndex]("robot_name", std::string("H1"));

  // Load policy paths from config
  policyPaths = config("policy_path", std::vector<std::string>{"walking_better_h1.onnx"});

  logTiming_ = config("log_timing");
  timingLogInterval_ = config("timing_log_interval");
  useQP = config("policies")[currentPolicyIndex]("use_QP", true); 
  useForceSensors = config("policies")[currentPolicyIndex]("use_force_sensors", false);
  useResidual = config("policies")[currentPolicyIndex]("use_residual", false);
  anchorLinks.clear();
  residualAnchorWeightJoints.clear();
  residualAnchorWeightJointIndices.clear();
  if (useForceSensors)
  {
    feetForceSensors = config("Robot")(robotName)("force_sensors");
    feetForceSensors.empty() ? mc_rtc::log::error_and_throw("please specify force_sensors in yaml for your Robot.") : mc_rtc::log::info("Using feet force sensors");
    feetForces = std::vector<Eigen::Vector3d>(feetForceSensors.size(), Eigen::Vector3d::Zero());
    oneFootForceRatio = 1.0 / static_cast<double>(feetForceSensors.size());

    for (const auto & sensorName : feetForceSensors)
    {
      if (!robot().hasForceSensor(sensorName))
        mc_rtc::log::error_and_throw("No force sensor named {} in {}", sensorName, robotName);
      anchorLinks.emplace_back(robot().forceSensor(sensorName).parentBody());
    }
  }
  if (useResidual)
  {
    const auto & confAnchorLinks = config("Robot")(robotName)("anchor_links", std::vector<std::string>{});
    if(!confAnchorLinks.empty())
    {
      anchorLinks = confAnchorLinks;
    }

    residualAnchorWeightJoints = config("Robot")(robotName)("knee_joints", std::vector<std::string>{});

    residualAnchorWeightJointIndices.reserve(residualAnchorWeightJoints.size());
    for(const auto & jointName : residualAnchorWeightJoints)
    {
      const int jointIndex = robot().jointIndexByName(jointName);
      if(jointIndex < 0)
      {
        mc_rtc::log::warning("[RLController] residual_anchor_weight_joints contains unknown joint {}", jointName);
        continue;
      }
      residualAnchorWeightJointIndices.push_back(jointIndex + 5);
    }

    if(!residualAnchorWeightJointIndices.empty() && residualAnchorWeightJointIndices.size() < anchorLinks.size())
    {
      mc_rtc::log::warning(
          "[RLController] residual_anchor_weight_joints has {} entries, but anchorLinks has {}. Falling back to geometric averaging for missing weights.",
          residualAnchorWeightJointIndices.size(), anchorLinks.size());
    }
  }
  isTorqueControl = config("policies")[currentPolicyIndex]("is_torque_control", false);
  if(isTorqueControl)
  {
    mc_rtc::log::info("Using Torque Control mode");
    datastore().make<std::string>("ControlMode", "Torque");
  }
  else
  {
    mc_rtc::log::info("Using Position Control mode");
    datastore().make<std::string>("ControlMode", "Position");
  }
 
  initializeRobot(config);
  initializeRLPolicy(config);
}

void RLController::switchPolicy(int policyIndex, const mc_rtc::Configuration & config)
{
  if(policyIndex < 0 || policyIndex >= static_cast<int>(policyPaths.size())) {
    mc_rtc::log::error("Invalid policy index: {}", policyIndex);
    return;
  }
  
  mc_rtc::log::info("Switching from policy [{}] to policy [{}]", currentPolicyIndex, policyIndex);
  currentPolicyIndex = policyIndex;
  robotName = config("policies")[currentPolicyIndex]("robot_name", std::string("H1"));
  
  // Update policy-specific boolean flags
  useResidual = config("policies")[currentPolicyIndex]("use_residual", false);
  useForceSensors = config("policies")[currentPolicyIndex]("use_force_sensors", false);
  anchorLinks.clear();
  residualAnchorWeightJoints.clear();
  residualAnchorWeightJointIndices.clear();
  if (useForceSensors)
  {
    feetForceSensors = config("Robot")(robotName)("force_sensors");
    feetForceSensors.empty() ? mc_rtc::log::error_and_throw("please specify force_sensors in yaml for your Robot.") : mc_rtc::log::info("Using feet force sensors");
    feetForces = std::vector<Eigen::Vector3d>(feetForceSensors.size(), Eigen::Vector3d::Zero());
    oneFootForceRatio = 1.0 / static_cast<double>(feetForceSensors.size());

    for (const auto & sensorName : feetForceSensors)
    {
      if (!robot().hasForceSensor(sensorName))
        mc_rtc::log::error_and_throw("No force sensor named {} in {}", sensorName, robotName);
      anchorLinks.emplace_back(robot().forceSensor(sensorName).parentBody());
    }
  }
  if (useResidual)
  {
    const auto & confAnchorLinks = config("Robot")(robotName)("anchor_links", std::vector<std::string>{});
    if(!confAnchorLinks.empty())
    {
      anchorLinks = confAnchorLinks;
    }

    residualAnchorWeightJoints = config("Robot")(robotName)("knee_joints", std::vector<std::string>{});

    residualAnchorWeightJointIndices.reserve(residualAnchorWeightJoints.size());
    for(const auto & jointName : residualAnchorWeightJoints)
    {
      const int jointIndex = robot().jointIndexByName(jointName);
      if(jointIndex < 0)
      {
        mc_rtc::log::warning("[RLController] residual_anchor_weight_joints contains unknown joint {}", jointName);
        continue;
      }
      residualAnchorWeightJointIndices.push_back(jointIndex + 5);
    }

    if(!residualAnchorWeightJointIndices.empty() && residualAnchorWeightJointIndices.size() < anchorLinks.size())
    {
      mc_rtc::log::warning(
          "[RLController] residual_anchor_weight_joints has {} entries, but anchorLinks has {}. Falling back to geometric averaging for missing weights.",
          residualAnchorWeightJointIndices.size(), anchorLinks.size());
    }
  }
  useQP = config("policies")[currentPolicyIndex]("use_QP", true);
  isTorqueControl = config("policies")[currentPolicyIndex]("is_torque_control", false);
  if(isTorqueControl) datastore().get<std::string>("ControlMode") = "Torque";
  else datastore().get<std::string>("ControlMode") = "Position";

  configRL(config);

  // Update PD gains
  pd_gains_ratio = config("policies")[currentPolicyIndex]("pd_gains_ratio", 1.0);
  std::map<std::string, double> kp = config("policies")[currentPolicyIndex]("kp");
  std::map<std::string, double> kd = config("policies")[currentPolicyIndex]("kd");

  for(size_t i = 0; i < dofNumber; ++i) {
    const auto & jName = robot().mb().joint(static_cast<int>(i + 1)).name();  // +1 to skip Root
    if(kp.count(jName)) {
      kp_vector(i) = kp[jName];
    }
    if(kd.count(jName)) {
      kd_vector(i) = kd[jName];
    }
  }
  // Update PD gains if necessary
  setPDGains(kp_vector, kd_vector);    
}

void RLController::tasksComputation(Eigen::VectorXd & currentTargetPosition)
{
  auto & real_robot = realRobot(robots()[0].name());

  auto q = real_robot.encoderValues();
  currentPos = Eigen::VectorXd::Map(q.data(), q.size());
  auto vel = real_robot.encoderVelocities();
  currentVel = Eigen::VectorXd::Map(vel.data(), vel.size());
  auto tau = real_robot.jointTorques();
  currentTau = Eigen::VectorXd::Map(tau.data(), tau.size());

  tau_d = (pd_gains_ratio * kp_vector).cwiseProduct(currentTargetPosition - currentPos) - (pd_gains_ratio * kd_vector).cwiseProduct(currentVel);
   
  size_t i = 0;
  for (const auto &joint_name : jointNames)
  {
    torque_target[joint_name][0] = tau_d[i];
    i++;
  }  
}

void RLController::updateRobotCmdAfterQP()
{
  qOut = robot().mbc().q;
  alphaOut = robot().mbc().alpha;
  tauOut = robot().mbc().jointTorque;

  floatingBase_qOut = rbd::paramToVector(robot().mb(), qOut);
  floatingBase_alphaOut = rbd::dofToVector(robot().mb(), alphaOut);
  floatingBase_tauOut = rbd::dofToVector(robot().mb(), tauOut);

  auto q = qOut;
  auto alpha = alphaOut;
  auto tau = tauOut;
  
  size_t i = 0;
  for (const auto &joint_name : jointNames)
  {
    q[robot().jointIndexByName(joint_name)][0] = q_cmd[i];
    alpha[robot().jointIndexByName(joint_name)][0] = 0.0;
    tau[robot().jointIndexByName(joint_name)][0] = tau_cmd[i];
    i++;
  }

  floatingBase_qOutPD = rbd::paramToVector(robot().mb(), q);
  floatingBase_alphaOutPD = rbd::dofToVector(robot().mb(), alpha);
  floatingBase_tauOutPD = rbd::dofToVector(robot().mb(), tau);

  // Update q and qdot for position control
  robot().mbc().q = q;
  robot().mbc().alpha = alpha; // For RL policy qdot ref = 0
  // Update joint torques for torque control
  robot().mbc().jointTorque = tau;
  // Both are always updated despite they are not used by the robot
  // They are still used by the QP
}

void RLController::computeInversePD()
{
  // Using QP:  
  ddot_qp_w_floatingBase = rbd::dofToVector(robot().mb(), robot().mbc().alphaD);
  ddot_qp = ddot_qp_w_floatingBase.tail(dofNumber); // Exclude the floating base part
  auto & real_robot = realRobot(robots()[0].name());

  rbd::ForwardDynamics fd(real_robot.mb());
  fd.computeH(real_robot.mb(), real_robot.mbc());
  fd.computeC(real_robot.mb(), real_robot.mbc());
  Eigen::MatrixXd M_w_floatingBase = fd.H();
  Eigen::VectorXd Cg_w_floatingBase = fd.C();

  Eigen::VectorXd extTorques = Eigen::VectorXd::Zero(robot().mb().nrDof());
  if (useResidual)
  {
    if (robotName == "H1")
    {
      auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
      extTorques = extTorqueSensor.torques();
    }
  }
  Eigen::VectorXd tau_cmd_w_floatingBase = M_w_floatingBase*ddot_qp_w_floatingBase + Cg_w_floatingBase - extTorques;
  tau_cmd = tau_cmd_w_floatingBase.tail(dofNumber);

  Eigen::VectorXd tau_rl_w_floating_base = Eigen::VectorXd::Zero(robot().mb().nrDof());
  tau_rl_w_floating_base.tail(dofNumber) = tau_rl;
  qddot_rl_simulatedMeasure = M_w_floatingBase.llt().solve(tau_rl_w_floating_base + extTorques - Cg_w_floatingBase).tail(dofNumber);
  qdot_rl_simulatedMeasure = currentVel + qddot_rl_simulatedMeasure*timeStep;
  q_rl_simulatedMeasure += qdot_rl_simulatedMeasure*timeStep;

  Eigen::MatrixXd Kp_inv = (pd_gains_ratio * kp_vector).cwiseInverse().asDiagonal();

  q_cmd = currentPos + Kp_inv*(tau_cmd + (pd_gains_ratio * kd_vector).cwiseProduct(currentVel)); // Inverse PD control to get the commanded position <=> RL position control
}

void RLController::computeRLStateSimulated()
{
  auto & real_robot = realRobot(robots()[0].name());
  rbd::ForwardDynamics fd(real_robot.mb());
  fd.computeH(real_robot.mb(), real_robot.mbc());
  fd.computeC(real_robot.mb(), real_robot.mbc());
  Eigen::MatrixXd M_w_floatingBase = fd.H();
  Eigen::VectorXd Cg_w_floatingBase = fd.C();

  Eigen::VectorXd extTorques = Eigen::VectorXd::Zero(robot().mb().nrDof());
  if (useResidual)
  {
    auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
    extTorques = extTorqueSensor.torques();
  }
  tau_rl = (pd_gains_ratio * kp_vector).cwiseProduct(q_rl - currentPos) - (pd_gains_ratio * kd_vector).cwiseProduct(currentVel);
  Eigen::VectorXd tau_rl_w_floating_base = Eigen::VectorXd::Zero(robot().mb().nrDof());
  tau_rl_w_floating_base.tail(dofNumber) = tau_rl;
  Eigen::VectorXd content = tau_rl_w_floating_base + extTorques - Cg_w_floatingBase; // Add the external torques to the desired torques

  qddot_rl_simulatedMeasure = M_w_floatingBase.llt().solve(content).tail(dofNumber);
  qdot_rl_simulatedMeasure = currentVel + qddot_rl_simulatedMeasure*timeStep;
  q_rl_simulatedMeasure = currentPos + qdot_rl_simulatedMeasure*timeStep;
  tau_err = tau_cmd - tau_rl;
  tau_err_norm = tau_err.norm();
  Eigen::VectorXd tau_err_w_floating_base = Eigen::VectorXd::Zero(robot().mb().nrDof());
  tau_err_w_floating_base.tail(dofNumber) = tau_err;
  qddot_err = M_w_floatingBase.llt().solve(tau_err_w_floating_base).tail(dofNumber);
  qddot_err_norm = qddot_err.norm();
}

void RLController::addLog()
{
  // Robot State variables
  logger().addLogEntry("RLController_refAccel", [this]() { return refAccel; });
  logger().addLogEntry("RLController_tau_d", [this]() { return tau_d; });
  logger().addLogEntry("RLController_kp_base", [this]() { return kp_vector; });
  logger().addLogEntry("RLController_kd_base", [this]() { return kd_vector; });
  logger().addLogEntry("RLController_kp_actual", [this]() { return current_kp; });
  logger().addLogEntry("RLController_kd_actual", [this]() { return current_kd; });
  logger().addLogEntry("RLController_pd_gains_ratio", [this]() { return pd_gains_ratio; });
  logger().addLogEntry("RLController_currentPos", [this]() { return currentPos; });
  logger().addLogEntry("RLController_currentVel", [this]() { return currentVel; });
  logger().addLogEntry("RLController_q_cmd", [this]() { return q_cmd; });
  logger().addLogEntry("RLController_qddot_qp", [this]() { return ddot_qp; });
  logger().addLogEntry("RLController_qddot_qp_w_floatingBase", [this]()
  { return ddot_qp_w_floatingBase; });
  logger().addLogEntry("RLController_tau_cmd", [this]() { return tau_cmd; });

  // RL variables
  logger().addLogEntry("RLController_RL_q", [this]() { return q_rl; });
  logger().addLogEntry("RLController_RL_qSimulatedMeasure", [this]() { return q_rl_simulatedMeasure; });
  logger().addLogEntry("RLController_RL_qdotSimulatedMeasure", [this]() { return qdot_rl_simulatedMeasure; });
  logger().addLogEntry("RLController_RL_qddotSimulatedMeasure", [this]() { return qddot_rl_simulatedMeasure; });
  logger().addLogEntry("RLController_tauRL", [this]() { return tau_rl; });
  logger().addLogEntry("RLController_pastAction", [this]() { return a_simuOrder; });
  logger().addLogEntry("RLController_qZero", [this]() { return q_zero_vector; });
  logger().addLogEntry("RLController_a_before", [this]() { return a_before_vector; });
  logger().addLogEntry("RLController_currentObservation", [this]() { return currentObservation_; });
  logger().addLogEntry("RLController_a_vector", [this]() { return a_vector; });
  logger().addLogEntry("RLController_a_simulationOrder", [this]() { return a_simuOrder; });
  logger().addLogEntry("RLController_currentAction", [this]() { return currentAction_; });
  logger().addLogEntry("RLController_latestAction", [this]() { return latestAction_; });
  logger().addLogEntry("RLController_baseAngVel", [this]() { return baseAngVel; });
  logger().addLogEntry("RLController_rpy", [this]() { return rpy; });
  logger().addLogEntry("RLController_legPos", [this]() { return legPos; });
  logger().addLogEntry("RLController_legVel", [this]() { return legVel; });
  logger().addLogEntry("RLController_legAction", [this]() { return legAction; });
  logger().addLogEntry("RLController_phase", [this]() { return phase_; });
  
  // Controller state variables
  logger().addLogEntry("RLController_useQP", [this]() { return useQP; });

  // RL Controller
  logger().addLogEntry("RLController_q_lim_upper", [this]() { return jointLimitsPos_upper; });
  logger().addLogEntry("RLController_q_lim_lower", [this]() { return jointLimitsPos_lower; });
  logger().addLogEntry("RLController_qdot_lim_upper", [this]() { return jointLimitsVel_upper; });
  logger().addLogEntry("RLController_qdot_lim_lower", [this]() { return jointLimitsVel_lower; });
  logger().addLogEntry("RLController_qdot_limHard_upper", [this]() { return jointLimitsHardVel_upper; });
  logger().addLogEntry("RLController_qdot_limHard_lower", [this]() { return jointLimitsHardVel_lower; });
  logger().addLogEntry("RLController_ankleDistanceNorm", [this]() { return ankleDistanceNorm; });
  logger().addLogEntry("RLController_tau_err", [this]() { return tau_err; });
  logger().addLogEntry("RLController_tau_err_norm", [this]() { return tau_err_norm; });
  logger().addLogEntry("RLController_qddot_err", [this]() { return qddot_err; }); 
  logger().addLogEntry("RLController_qddot_err_norm", [this]() { return qddot_err_norm; });
  std::vector<double> qOut_vec(robot().refJointOrder().size(), 0);
  logger().addLogEntry("RLController_qOutNoModification", [this, qOut_vec]() mutable -> const std::vector<double> &
                         {
                           auto & robot = this->robot();
                           for(size_t i = 0; i < qOut_vec.size(); ++i)
                           {
                             auto mbcIndex = robot.jointIndexInMBC(i);
                             if(mbcIndex != -1) { qOut_vec[i] = qOut[static_cast<size_t>(mbcIndex)][0]; }
                           }
                           return qOut_vec;
                         });
  std::vector<double> alphaOut_vec(robot().refJointOrder().size(), 0);
  logger().addLogEntry("RLController_alphaOutNoModification", [this, alphaOut_vec]() mutable -> const std::vector<double> &
                         {
                           auto & robot = this->robot();
                           for(size_t i = 0; i < alphaOut_vec.size(); ++i)
                           {
                             auto mbcIndex = robot.jointIndexInMBC(i);
                             if(mbcIndex != -1) { alphaOut_vec[i] = alphaOut[static_cast<size_t>(mbcIndex)][0]; }
                           }
                           return alphaOut_vec;
                         });
  std::vector<double> tauOut_vec(robot().refJointOrder().size(), 0);
  logger().addLogEntry("RLController_tauOutNoModification", [this, tauOut_vec]() mutable -> const std::vector<double> &
                         {
                           auto & robot = this->robot();
                           for(size_t i = 0; i < tauOut_vec.size(); ++i)
                           {
                             auto mbcIndex = robot.jointIndexInMBC(i);
                             if(mbcIndex != -1) { tauOut_vec[i] = tauOut[static_cast<size_t>(mbcIndex)][0]; }
                           }
                           return tauOut_vec;
                         });

  logger().addLogEntry("RLController_floatingBase_qOutNoModification", [this]() { return floatingBase_qOut; });
  logger().addLogEntry("RLController_floatingBase_alphaOutNoModification", [this]() { return floatingBase_alphaOut; });
  logger().addLogEntry("RLController_floatingBase_tauOutNoModification", [this]() { return floatingBase_tauOut; });
  logger().addLogEntry("RLController_floatingBase_qOutPD", [this]() { return floatingBase_qOutPD; });
  logger().addLogEntry("RLController_floatingBase_alphaOutPD", [this]() { return floatingBase_alphaOutPD; });
  logger().addLogEntry("RLController_floatingBase_tauOutPD", [this]() { return floatingBase_tauOutPD; });
  logger().addLogEntry("RLController_floatingBase_qIn", [this]() { return floatingBase_qIn; });
  logger().addLogEntry("RLController_floatingBase_alphaIn", [this]() { return floatingBase_alphaIn; });
  
  // Log current policy (combined index and path)
  logger().addLogEntry("RLController_currentPolicy", [this]() { 
    return std::to_string(currentPolicyIndex) + ": " + policyPaths[currentPolicyIndex]; 
  });
}

void RLController::addGui(const mc_rtc::Configuration & config)
{
  gui()->addElement({"RLController", "Options"},
  mc_rtc::gui::Checkbox("Compensate External Forces", compensateExternalForces));
  // Add a button to change the velocity command
  gui()->addElement({"RLController", "Policy"},
  mc_rtc::gui::ArrayInput("Velocity Command RL", {"X", "Y", "Yaw"}, velCmdRL_));
  // Add a button to change the speed multiplier for joystick
  gui()->addElement({"RLController", "Policy"},
  mc_rtc::gui::NumberInput("Max Vel via Joystick", maxVelCmd));
  gui()->addElement({"RLController", "Policy"},
  mc_rtc::gui::NumberInput("Max yaw via Joystick", maxYawCmd));

  // Add a dropdown to select policy
  gui()->addElement(
    {"RLController", "Policy"},
    mc_rtc::gui::Label("Current policy", [this]() -> const std::string & { 
      return policyPaths[currentPolicyIndex]; 
    }),
    mc_rtc::gui::ComboInput(
      "Select policy",
      policyPaths,
      [this]() -> const std::string & { 
        return policyPaths[currentPolicyIndex]; 
      },
      [this, config](const std::string & selected) {  // Capture config by VALUE (makes a safe copy)
        // Find the index of the selected policy
        auto it = std::find(policyPaths.begin(), policyPaths.end(), selected);
        if(it != policyPaths.end()) {
          int newIndex = static_cast<int>(std::distance(policyPaths.begin(), it));
          mc_rtc::log::info("User requested policy switch to [{}]: {}", newIndex, selected);
          // Switch to new policy without reinitializing robot
          switchPolicy(newIndex, config);
        }
      }
    )
  );
  // Add a button to reload the current policy
  gui()->addElement(
    {"RLController", "Policy"},
    mc_rtc::gui::Button("Reload current policy", [this, config]() {
      mc_rtc::log::info("User requested to reload current policy [{}]", currentPolicyIndex);
      switchPolicy(currentPolicyIndex, config);
    })
  );

  // Add PD gains ratio slider
  gui()->addElement({"RLController", "PD Gains"},
    mc_rtc::gui::NumberSlider("PD Gains Ratio", [this]() { return pd_gains_ratio; },
      [this](double v) { 
        pd_gains_ratio = v;
        // Update the actual gains on the robot when ratio changes
        if(datastore().has(robot().name() + "::SetPDGains"))
          setPDGains(kp_vector, kd_vector);
        else
          mc_rtc::log::warning("Cannot set PD gains ratio, SetPDGains not found in datastore");
      }, 0.0, 2.0)
  );
  
  // Display actual gains (base * ratio) for each joint - read-only
  const auto & jointOrder = robot().refJointOrder();
  for(size_t i = 0; i < jointOrder.size() && i < static_cast<size_t>(kp_vector.size()); ++i) {
    gui()->addElement({"RLController", "PD Gains", "Actual Kp"},
      mc_rtc::gui::Label(jointOrder[i], [this, i]() { 
        return std::to_string(pd_gains_ratio * kp_vector(i)); 
      })
    );
    gui()->addElement({"RLController", "PD Gains", "Actual Kd"},
      mc_rtc::gui::Label(jointOrder[i], [this, i]() { 
        return std::to_string(pd_gains_ratio * kd_vector(i)); 
      })
    );
  }

  gui()->addElement({"RLController", "Options"},
    mc_rtc::gui::Transform("Anchor Frame",contact_anchor_tf)
  );
}

void RLController::initializeRobot(const mc_rtc::Configuration & config)
{
  // get the joints order (urdf) depending on the robot used
  robotName = config("policies")[currentPolicyIndex]("robot_name", std::string("H1"));
  auto mcRtcJointsOrder = config("Robot")(robotName)("mc_rtc_joints_order");

  dofNumber = robot().mb().nrDof() - 6; // Remove the floating base part (6 DoF)

  auto & real_robot = realRobot(robots()[0].name());

  if (useResidual)
  {
    if(robotName == "H1")
    {
      leftAnklePos = real_robot.collisionTransform("left_ankle_link").translation();
      rightAnklePos = real_robot.collisionTransform("right_ankle_link").translation();
      ankleDistanceNorm = (leftAnklePos - rightAnklePos).norm();
    }
  }
  jointLimitsHardPos_upper = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardPos_lower = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardVel_upper = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardVel_lower = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardTau_upper = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardTau_lower = Eigen::VectorXd::Zero(dofNumber);

  for (size_t i = 0 ; i < robot().refJointOrder().size() ; i++)
  {
    const std::string & jname = robot().refJointOrder()[i];
    auto mcJointId = robot().jointIndexByName(jname);
    if (robot().mbc().q[mcJointId].empty())
      continue;
    
    jointLimitsHardPos_lower(i) = robot().ql().at(mcJointId)[0];
    jointLimitsHardPos_upper(i) = robot().qu().at(mcJointId)[0];
    jointLimitsHardVel_lower(i) = robot().vl().at(mcJointId)[0];
    jointLimitsHardVel_upper(i) = robot().vu().at(mcJointId)[0];
    jointLimitsHardTau_lower(i) = robot().tl().at(mcJointId)[0];
    jointLimitsHardTau_upper(i) = robot().tu().at(mcJointId)[0];
  }

  jointLimitsPos_upper = jointLimitsHardPos_upper - (jointLimitsHardPos_upper - jointLimitsHardPos_lower)*dsPercent;
  jointLimitsPos_lower = jointLimitsHardPos_lower + (jointLimitsHardPos_upper - jointLimitsHardPos_lower)*dsPercent;

  jointLimitsVel_upper = jointLimitsHardVel_upper*velPercent;
  jointLimitsVel_lower = jointLimitsHardVel_lower*velPercent;

  mc_rtc::log::info("[RLController] Joint limits pos upper: {}", jointLimitsPos_upper.transpose());
  mc_rtc::log::info("[RLController] Joint limits pos lower: {}", jointLimitsPos_lower.transpose());
  mc_rtc::log::info("[RLController] Joint limits vel upper: {}", jointLimitsVel_upper.transpose());
  mc_rtc::log::info("[RLController] Joint limits vel lower: {}", jointLimitsVel_lower.transpose());
  refAccel = Eigen::VectorXd::Zero(dofNumber); // TVM
  q_rl = Eigen::VectorXd::Zero(dofNumber);
  q_rl_simulatedMeasure = Eigen::VectorXd::Zero(dofNumber);
  qdot_rl_simulatedMeasure = Eigen::VectorXd::Zero(dofNumber);
  qddot_rl_simulatedMeasure = Eigen::VectorXd::Zero(dofNumber);
  tau_rl = Eigen::VectorXd::Zero(dofNumber);
  q_zero_vector = Eigen::VectorXd::Zero(dofNumber);
  tau_d = Eigen::VectorXd::Zero(dofNumber);
  kp_vector = Eigen::VectorXd::Zero(dofNumber);
  kd_vector = Eigen::VectorXd::Zero(dofNumber);
  currentPos = Eigen::VectorXd::Zero(dofNumber);
  currentVel = Eigen::VectorXd::Zero(dofNumber);
  currentTau = Eigen::VectorXd::Zero(dofNumber);

  ddot_qp = Eigen::VectorXd::Zero(dofNumber); // Desired acceleration in the QP solver
  ddot_qp_w_floatingBase = Eigen::VectorXd::Zero(robot().mb().nrDof()); // Desired acceleration in the QP solver with floating base
  
  tau_cmd = Eigen::VectorXd::Zero(dofNumber); // Final torque that control the robot
  q_cmd = Eigen::VectorXd::Zero(dofNumber); // The commended position send to the internal PD of the robot
  tau_err = Eigen::VectorXd::Zero(dofNumber);
  qddot_err = Eigen::VectorXd::Zero(dofNumber);
  
  // Get the gains from the configuration or set default values
  pd_gains_ratio = config("policies")[currentPolicyIndex]("pd_gains_ratio", 1.0);
  std::map<std::string, double> kp = config("policies")[currentPolicyIndex]("kp");
  std::map<std::string, double> kd = config("policies")[currentPolicyIndex]("kd");
  // Get the default posture target from the robot's posture task
  std::shared_ptr<mc_tasks::PostureTask> FSMPostureTask = getPostureTask(robot().name());
  auto posture = FSMPostureTask->posture();
  size_t i = 0;
  std::vector<std::string> joint_names;
  joint_names.reserve(robot().mb().joints().size());
  
  for (const auto &j : robot().mb().joints()) {
      const std::string &joint_name = j.name();
      if(j.type() == rbd::Joint::Type::Rev)
      {
        jointNames.emplace_back(joint_name);  
        if (const auto &t = posture[robot().jointIndexByName(joint_name)]; !t.empty()) {
            kp_vector[i] = kp.at(joint_name);
            kd_vector[i] = kd.at(joint_name);
            q_rl[i] = t[0];
            q_zero_vector[i] = t[0];
            torque_target[joint_name] = {0.0};
            mc_rtc::log::info("[RLController] Joint {}: currentTargetPosition {}, kp {}, kd {}", joint_name, q_rl[i], kp_vector[i], kd_vector[i]);
            i++;
        }
      }
  }
  current_kp = kp_vector;
  current_kd = kd_vector;
  solver().removeTask(FSMPostureTask);
  if(!datastore().has("anchorFrameFunction") && (useResidual || useForceSensors))
  {
    datastore().make_call("anchorFrameFunction", [this, &config](const mc_rbdyn::Robot & real_robot) {return createContactAnchor(real_robot, config);});
  }

  // State after QP without any modification
  qOut = robot().mbc().q;
  alphaOut = robot().mbc().alpha;
  tauOut = robot().mbc().jointTorque;

  floatingBase_qIn = Eigen::VectorXd::Zero(robot().mb().nrParams());
  floatingBase_alphaIn = Eigen::VectorXd::Zero(robot().mb().nrDof());
  floatingBase_qOut = Eigen::VectorXd::Zero(robot().mb().nrParams());
  floatingBase_alphaOut = Eigen::VectorXd::Zero(robot().mb().nrDof());
  floatingBase_tauOut = Eigen::VectorXd::Zero(robot().mb().nrDof());
  floatingBase_qOutPD = Eigen::VectorXd::Zero(robot().mb().nrParams());
  floatingBase_alphaOutPD = Eigen::VectorXd::Zero(robot().mb().nrDof());
  floatingBase_tauOutPD = Eigen::VectorXd::Zero(robot().mb().nrDof());

  floatingBase_qOut = rbd::paramToVector(robot().mb(), qOut);
  floatingBase_alphaOut = rbd::dofToVector(robot().mb(), alphaOut);
  floatingBase_tauOut = rbd::paramToVector(robot().mb(), tauOut);
  floatingBase_qOutPD = floatingBase_qOut;
  floatingBase_alphaOutPD = floatingBase_alphaOut;
  floatingBase_tauOutPD = floatingBase_tauOut;
  auto qIn = real_robot.mbc().q;
  auto alphaIn = real_robot.mbc().alpha;
  floatingBase_qIn = rbd::paramToVector(robot().mb(), qIn);
  floatingBase_alphaIn = rbd::dofToVector(robot().mb(), alphaIn);
}

void RLController::initializeRLPolicy(const mc_rtc::Configuration & config)
{
  // load policy specific configuration

  configRL(config);
  auto & real_robot = realRobot(robots()[0].name());
  
  std::string baseName;
  if(robotName == "H1")
    baseName = "pelvis";
  else if (robotName == "HRP5P")
    baseName = "Body";
  else if(robotName == "Go2")
    baseName = "base";
  else
    baseName = "root";

  projected_gravity = Eigen::Vector3d::Zero();
  baseAngVel = real_robot.bodyVelW(baseName).angular();
  baseLinVel = real_robot.bodyVelW(baseName).linear();

  Eigen::Matrix3d baseRot = real_robot.bodyPosW(baseName).rotation();
  rpy = mc_rbdyn::rpyFromMat(baseRot);
    
  mc_rtc::log::info("[RLController] Posture target initialized with {} joints", dofNumber); 

  // Initialize reference position and last actions for action blending
  size_t vect_size = usedJoints_mcRtcOrder.size(); 
  a_before_vector = Eigen::VectorXd::Zero(dofNumber);
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

  mc_rtc::log::info("Reference position initialized with {} joints", q_zero_vector.size());
  q_rl = q_zero_vector;  // Start with reference position
  
  currentAction_ = Eigen::VectorXd::Zero(dofNumber);
  latestAction_ = Eigen::VectorXd::Zero(dofNumber);
  
  // Initialize new observation components
  velCmdRL_ = Eigen::Vector3d::Zero();  // Default command (x, y, yaw)
  phase_ = 0.0;  // Phase for periodic gait
  startPhase_ = std::chrono::steady_clock::now();  // For phase calculation
}

void RLController::configRL(const mc_rtc::Configuration & config)
{
  mc_rtc::log::info("Loading RL policy [{}]: {}", currentPolicyIndex, policyPaths[currentPolicyIndex]);
  try {
    rlPolicy_ = std::make_unique<RLPolicyInterface>(policyPaths[currentPolicyIndex]);
    if(rlPolicy_) {
      mc_rtc::log::success("RL policy loaded successfully");
      // Initialize observation vector with the correct size from the loaded policy
      currentObservation_ = Eigen::VectorXd::Zero(rlPolicy_->getObservationSize());
      mc_rtc::log::info("Initialized observation vector with size: {}", rlPolicy_->getObservationSize());
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
    policySimulatorHandling_ = std::make_unique<PolicySimulatorHandling>();
  }
  else
  {
    mc_rtc::log::info("Using {} handling", simulator);
    policySimulatorHandling_ = std::make_unique<PolicySimulatorHandling>(simulator, robotName);
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
    usedJoints_simuOrder = policySimulatorHandling_->getSimulatorIndices(usedJoints_mcRtcOrder);
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
    usedJoints_simuOrder = std::vector<int>(dofNumber);
    std::iota(usedJoints_simuOrder.begin(), usedJoints_simuOrder.end(), 0);
  }
  maxVelCmd = config("policies")[currentPolicyIndex]("speed_multiplier_joystick", 0.6);
  maxYawCmd = config("policies")[currentPolicyIndex]("max_yaw_joystick", 0.7);

  actionScale = config("policies")[currentPolicyIndex]("action_scale", 1.0);
  policyPeriodMs = config("policies")[currentPolicyIndex]("policy_period_ms", 20.0);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> RLController::getPDGains()
{
  std::string robot_name = robot().name();
  std::vector<double> proportionalGains_vec(kp_vector.data(), kp_vector.data() + kp_vector.size());
  std::vector<double> dampingGains_vec(kd_vector.data(), kd_vector.data() + kd_vector.size());
  datastore().call<bool>(robot_name + "::GetPDGains", proportionalGains_vec, dampingGains_vec);
  Eigen::VectorXd p_vec = Eigen::VectorXd::Map(proportionalGains_vec.data(), proportionalGains_vec.size());
  Eigen::VectorXd d_vec = Eigen::VectorXd::Map(dampingGains_vec.data(), dampingGains_vec.size());
  return std::make_tuple(p_vec, d_vec);
}

bool RLController::setPDGains(Eigen::VectorXd p_vec, Eigen::VectorXd d_vec)
{
  // Set low gains for RL if necesssary
  if(!(datastore().has(robot().name() + "::GetPDGains")))
  {
    mc_rtc::log::error("PD gains can't be changed, GetPDGains not found in datastore.");
    return false;
  }
  if(!gainsUpdateRequired())
  {
    mc_rtc::log::info("PD gains are already up to date, no change needed.");
    return true;
  }
  std::string robot_name = robot().name();
  // Apply ratio to get actual gains
  Eigen::VectorXd actual_kp = pd_gains_ratio * p_vec;
  Eigen::VectorXd actual_kd = pd_gains_ratio * d_vec;

  // Update kp and kd use by the robot or simulator (Internal PD)
  const std::vector<double> proportionalGains_vec(actual_kp.data(), actual_kp.data() + actual_kp.size());
  const std::vector<double> dampingGains_vec(actual_kd.data(), actual_kd.data() + actual_kd.size());

  bool res = datastore().call<bool>(robot_name + "::SetPDGains", proportionalGains_vec, dampingGains_vec);
  if(res)
  {
    mc_rtc::log::info("[RLController] New PD Gains have been successfully updated");
    // Cache the gains we're setting
    current_kp = actual_kp;
    current_kd = actual_kd;
  }
  else
  {
    mc_rtc::log::error("Failed to set PD gains for {}", robot_name);
  }
  return res;
}

bool RLController::gainsUpdateRequired(double tol)
{
  // Get current gains from robot and update cache
  std::tie(current_kp, current_kd) = getPDGains();
  // Check if the current gains are close to the target gains (with ratio)
  Eigen::VectorXd target_kp = pd_gains_ratio * kp_vector;
  Eigen::VectorXd target_kd = pd_gains_ratio * kd_vector;
  bool gainCorrectionRequired = ((current_kp - target_kp).norm() >= tol) || ((current_kd - target_kd).norm() >= tol);
  if(gainCorrectionRequired)
  {
    mc_rtc::log::info("[RLController] gains correction needed: {}", gainCorrectionRequired);
    mc_rtc::log::info("[RLController] current_kp: {}", current_kp.transpose());
    mc_rtc::log::info("[RLController] current_kd: {}", current_kd.transpose());
    mc_rtc::log::info("[RLController] target_kp (ratio {}): {}", pd_gains_ratio, target_kp.transpose());
    mc_rtc::log::info("[RLController] target_kd (ratio {}): {}", pd_gains_ratio, target_kd.transpose());
  }
  return gainCorrectionRequired;
}

void RLController::initializeState()
{
  // Update PD gains if necessary
  setPDGains(kp_vector, kd_vector);
  tasksComputation(q_rl);
}

std::pair<sva::PTransformd, Eigen::Vector3d> RLController::createContactAnchor(const mc_rbdyn::Robot & anchorRobot, const mc_rtc::Configuration & config)
{
  double anchorLeftFootRatio = 0.5;
  bool hasForceBasedRatio = false;
  bool hasResidualBasedWeights = false;
  std::vector<double> residualAnchorWeights;

  Eigen::Vector3d weightedAnchor = Eigen::Vector3d::Zero();
  Eigen::Vector3d weightedVel = Eigen::Vector3d::Zero();
  double totalFootForceZ = 0.0;
  if(useForceSensors && !feetForceSensors.empty())
  {
    if(feetForces.size() != feetForceSensors.size())
    {
      feetForces = std::vector<Eigen::Vector3d>(feetForceSensors.size(), Eigen::Vector3d::Zero());
    }

    for(size_t i = 0; i < feetForceSensors.size(); ++i)
    {
      const auto & sensorName = feetForceSensors[i];
      const auto & forceSensor = anchorRobot.forceSensor(sensorName);
      const auto wrench = forceSensor.worldWrenchWithoutGravity(anchorRobot);
      const auto force = wrench.force();
      feetForces[i] = force;
      const double forceZ = std::max(0.0, force.z());
      totalFootForceZ += forceZ;
      weightedAnchor += forceZ * anchorRobot.bodyPosW(forceSensor.parentBody()).translation();
      weightedVel += forceZ * anchorRobot.bodyVelW(forceSensor.parentBody()).linear();
    }
  }

  if(totalFootForceZ > minFootForceForAnchor)
  {
    contact_anchor_tf = sva::PTransformd(Eigen::Matrix3d::Identity(), weightedAnchor / totalFootForceZ);
    oneFootForceRatio = 1.0 / static_cast<double>(feetForceSensors.size());
    hasForceBasedRatio = true;
    return {contact_anchor_tf, weightedVel / totalFootForceZ};
  }

  if(!hasForceBasedRatio && useResidual)
  {
    auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
    const auto & extTorques = extTorqueSensor.torques();
    const size_t weightedContacts = std::min(anchorLinks.size(), residualAnchorWeightJointIndices.size());
    if(weightedContacts > 0)
    {
      residualAnchorWeights.assign(weightedContacts, 0.0);
      double totalResidualTorque = 0.0;
      for(size_t i = 0; i < weightedContacts; ++i)
      {
        const int jointIndex = residualAnchorWeightJointIndices[i];
        if(jointIndex < 0 || jointIndex >= static_cast<int>(extTorques.size()))
        {
          continue;
        }
        const double tauExt = std::abs(extTorques[jointIndex]);
        residualAnchorWeights[i] = tauExt;
        totalResidualTorque += tauExt;
      }

      if(totalResidualTorque > minResidualTorqueForAnchor)
      {
        for(auto & w : residualAnchorWeights)
        {
          w /= totalResidualTorque;
        }
        hasResidualBasedWeights = true;
        if(residualAnchorWeights.size() == 2)
        {
          anchorLeftFootRatio = residualAnchorWeights[1];
        }
      }
    }
  }

  std::vector<Eigen::Vector3d> anchorPositions;
  std::vector<Eigen::Vector3d> anchorVels;
  anchorPositions.reserve(anchorLinks.size());
  anchorVels.reserve(anchorLinks.size());

  for(const auto & linkName : anchorLinks)
  {
    anchorPositions.push_back(anchorRobot.bodyPosW(linkName).translation());
    anchorVels.push_back(anchorRobot.bodyVelW(linkName).linear());
  }

  if(anchorPositions.empty())
  {
    mc_rtc::log::warning("[RLController] No anchorLinks available, please specify them in the controller config");
    return {contact_anchor_tf, Eigen::Vector3d::Zero()};
  }

  if(hasResidualBasedWeights && residualAnchorWeights.size() == anchorPositions.size())
  {
    Eigen::Vector3d contact_anchor = Eigen::Vector3d::Zero();
    Eigen::Vector3d anchor_vel = Eigen::Vector3d::Zero();
    for(size_t i = 0; i < anchorPositions.size(); ++i)
    {
      contact_anchor += residualAnchorWeights[i] * anchorPositions[i];
      anchor_vel += residualAnchorWeights[i] * anchorVels[i];
    }
    contact_anchor_tf = sva::PTransformd(Eigen::Matrix3d::Identity(), contact_anchor);
    return {contact_anchor_tf, anchor_vel};
  }

  if(anchorPositions.size() == 2)
  {
    oneFootForceRatio = anchorLeftFootRatio;
    const Eigen::Vector3d contact_anchor = anchorPositions[0] * (1 - oneFootForceRatio) + anchorPositions[1] * oneFootForceRatio;
    const Eigen::Vector3d anchor_vel = anchorVels[0] * (1 - oneFootForceRatio) + anchorVels[1] * oneFootForceRatio;
    contact_anchor_tf = sva::PTransformd(Eigen::Matrix3d::Identity(), contact_anchor);
    return {contact_anchor_tf, anchor_vel};
  }

  oneFootForceRatio = 1.0 / static_cast<double>(anchorPositions.size());
  Eigen::Vector3d contact_anchor = Eigen::Vector3d::Zero();
  Eigen::Vector3d anchor_vel = Eigen::Vector3d::Zero();
  for(size_t i = 0; i < anchorPositions.size(); ++i)
  {
    contact_anchor += anchorPositions[i];
    anchor_vel += anchorVels[i];
  }
  contact_anchor /= static_cast<double>(anchorPositions.size());
  anchor_vel /= static_cast<double>(anchorPositions.size());
  contact_anchor_tf = sva::PTransformd(Eigen::Matrix3d::Identity(), contact_anchor);

  return {contact_anchor_tf, anchor_vel};
}
