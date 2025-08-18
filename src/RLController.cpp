#include "RLController.h"
#include <Eigen/src/Core/VectorBlock.h>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <mc_rtc/logging.h>
#include <mc_rbdyn/configuration_io.h>
#include <chrono>
#include <cmath>


RLController::RLController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(rm, dt, config, Backend::TVM)
{
  logTiming_ = config("log_timing");
  timingLogInterval_ = config("timing_log_interval");

  //Initialize Constraints
  selfCollisionConstraint->setCollisionsDampers(solver(), {1.2, 400.0});
  solver().removeConstraintSet(dynamicsConstraint);
  dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(
    new mc_solver::DynamicsConstraint(robots(), 0, {0.1, 0.01, 0.0, 1.2, 400.0}, 0.9, true));
    // new mc_solver::DynamicsConstraint(robots(), 0, timeStep, {0.1, 0.01, 0.5}, 0.9, true, true));
  solver().addConstraintSet(dynamicsConstraint);

  // Initialize Tasks
  FDTask = std::make_shared<mc_tasks::PostureTask>(solver(), robot().robotIndex(), 0.0, 1000.0);
  FDTask->stiffness(0.0);
  FDTask->damping(0.0);
  FDTask->refAccel(refAccel);

  torqueTask = std::make_shared<mc_tasks::TorqueTask>(solver(), robot().robotIndex());

  initializeRobot(config);
  initializeRLPolicy(config);
  
  if(useAsyncInference_)
  {
    auto & ctl = *this;
    utils_.startInferenceThread(ctl);
  }

  addGui();
  addLog();
  mc_rtc::log::success("RLController init");
}

bool RLController::run()
{
  bool run = mc_control::fsm::Controller::run(mc_solver::FeedbackType::ClosedLoopIntegrateReal);
    auto ctrl_mode = datastore().get<std::string>("ControlMode");
  if (ctrl_mode.compare("Position") == 0)
    return positionControl(run);
  return torqueControl(run); // = ctrl_mode.compare("Torque") == 0 :
}

void RLController::reset(const mc_control::ControllerResetData & reset_data)
{
  mc_control::fsm::Controller::reset(reset_data);
  mc_rtc::log::success("RLController reset completed");
  // utils_.stopInferenceThread();
}

void RLController::tasksComputation(Eigen::VectorXd & currentTargetPosition)
{
  auto & robot = robots()[0];
  auto & real_robot = realRobot(robots()[0].name());

  auto q = robot.encoderValues();
  currentPos = Eigen::VectorXd::Map(q.data(), q.size());
  auto vel = robot.encoderVelocities();
  currentVel = Eigen::VectorXd::Map(vel.data(), vel.size());

  if(controlledByRL) tau_d = kp_vector.cwiseProduct(currentTargetPosition - currentPos) + kd_vector.cwiseProduct(-currentVel);
  else tau_d = high_kp_vector.cwiseProduct(currentTargetPosition - currentPos) + high_kd_vector.cwiseProduct(-currentVel);

  switch (taskType)
  {
    case TORQUE_TASK: // Torque Task
    {
      size_t i = 0;
      for (const auto &joint_name : jointNames)
      {
        torque_target[joint_name][0] = tau_d[i];
        i++;
      }
      break;
    }
    case FD_TASK: // Forward Dynamics Task
    {
      rbd::ForwardDynamics fd(real_robot.mb());
      fd.computeH(real_robot.mb(), real_robot.mbc());
      fd.computeC(real_robot.mb(), real_robot.mbc());
      Eigen::MatrixXd M_w_floatingBase = fd.H();
      Eigen::VectorXd Cg_w_floatingBase = fd.C();
      // Eigen::MatrixXd M = M_w_floatingBase.bottomRightCorner(dofNumber, dofNumber);
      // Eigen::VectorXd Cg = Cg_w_floatingBase.tail(dofNumber);
      
      auto extTorqueSensor = robot.device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
      // Eigen::VectorXd externalTorques = extTorqueSensor.torques().tail(dofNumber); // Exclude the floating base part
      Eigen::VectorXd tau_d_w_floating_base = Eigen::VectorXd::Zero(robot.mb().nrDof());
      tau_d_w_floating_base.tail(dofNumber) = tau_d.tail(dofNumber);
      Eigen::VectorXd content = tau_d_w_floating_base - Cg_w_floatingBase; // Add the external torques to the desired torques
      if(!compensateExternalForces) content += extTorqueSensor.torques();
      Eigen::VectorXd refAccel_w_floating_base = M_w_floatingBase.llt().solve(content);
      refAccel = refAccel_w_floating_base.tail(dofNumber); // Exclude the floating base part

      
      // Eigen::VectorXd content = tau_d - Cg; // Add the external torques to the desired torques
      // if(!compensateExternalForces) content += externalTorques;
      // refAccel = M.llt().solve(content);
      break;
    }
    default:
      mc_rtc::log::error("Invalid task type: {}", taskType);
      return;
  }
}

bool RLController::positionControl(bool run)
{
  robot().forwardKinematics();
  robot().forwardVelocity();
  robot().forwardAcceleration();
  auto q = robot().mbc().q;
  auto alpha = robot().mbc().alpha;

  if(!useQP) // Send RL Position directly
  {
    size_t i = 0;
    for (const auto &joint_name : jointNames)
    {
      q[robot().jointIndexByName(joint_name)][0] = q_rl_vector[i];
      alpha[robot().jointIndexByName(joint_name)][0] = 0.0;
      i++;
    }
    robot().mbc().q = q;
    robot().mbc().alpha = alpha; // Set velocity reference to zero
    
    a_before_vector = q_zero_vector - q_rl_vector; //no QP, rl vector is enough
    return true;
  }

  // Using QP (TorqueTask or ForwardDynamics Task):  
  rbd::paramToVector(robot().mbc().alphaD, ddot_qp_w_floatingBase);
  ddot_qp = ddot_qp_w_floatingBase.tail(dofNumber); // Exclude the floating base part

  // Use robot instead of realrobot because we are after the QP
  rbd::ForwardDynamics fd(robot().mb());
  fd.computeH(robot().mb(), robot().mbc());
  fd.computeC(robot().mb(), robot().mbc());
  Eigen::MatrixXd M_w_floatingBase = fd.H();
  Eigen::VectorXd Cg_w_floatingBase = fd.C();
  Eigen::MatrixXd M = M_w_floatingBase.bottomRightCorner(dofNumber, dofNumber);
  Eigen::VectorXd Cg = Cg_w_floatingBase.tail(dofNumber);

  Eigen::MatrixXd Kp_inv = current_kp.cwiseInverse().asDiagonal();
  auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
  Eigen::VectorXd externalTorques = extTorqueSensor.torques().tail(dofNumber);

  tau_qp = M*ddot_qp + Cg - externalTorques;
  tau_qp_w_floatingBase.tail(dofNumber) = tau_qp;
  
  q_cmd = currentPos + Kp_inv*(tau_qp + current_kd.cwiseProduct(currentVel)); // Inverse PD control to get the commanded position <=> RL position control

  tau_cmd_after_pd = current_kp.cwiseProduct(q_cmd - currentPos) - current_kd.cwiseProduct(currentVel); // PD control to get the commanded position after PD control
  auto tau = robot().mbc().jointTorque;
  
  size_t i = 0;
  for (const auto &joint_name : jointNames)
  {
    q[robot().jointIndexByName(joint_name)][0] = q_cmd[i];
    alpha[robot().jointIndexByName(joint_name)][0] = 0.0;
    tau[robot().jointIndexByName(joint_name)][0] = tau_cmd_after_pd[i];
    i++;
  }
  if(controlledByRL) robot().mbc().alpha = alpha; // Update the mbc with the new velocity to respect the RL policy
  robot().mbc().q = q; // Update the mbc with the new position
  // To close the loop on the external torques:
  robot().mbc().jointTorque = tau; // Update the mbc with the new torque
  a_before_vector = q_zero_vector - q_cmd;
  return run;
}

bool RLController::torqueControl(bool run)
{
  if (!useQP) // Compute RL torque
  {
    auto tau = robot().mbc().jointTorque;
    tau_d = kp_vector.cwiseProduct(q_rl_vector - currentPos) - kd_vector.cwiseProduct(currentVel);
    
    size_t i = 0;
    for (const auto &joint_name : jointNames)
    {
      tau[robot().jointIndexByName(joint_name)][0] = tau_d[i];
      i++;
    }

    robot().mbc().jointTorque = tau;
    a_before_vector = q_zero_vector - q_rl_vector; //no QP, rl vector is enough
    return true;
  }
  auto & robot = robots()[0];
  auto q = robot.encoderValues(); 
  currentPos = Eigen::VectorXd::Map(q.data(), q.size());

  a_before_vector = q_zero_vector - currentPos;

  return run;
}

void RLController::addLog()
{
  // Robot State variables
  logger().addLogEntry("RLController_refAccel", [this]() { return refAccel; });
  logger().addLogEntry("RLController_currentTargetPosition", [this]() { return q_rl_vector; });
  logger().addLogEntry("RLController_tau_d", [this]() { return tau_d; });
  logger().addLogEntry("RLController_kp", [this]() { return current_kp; });
  logger().addLogEntry("RLController_kd", [this]() { return current_kd; });
  logger().addLogEntry("RLController_currentPos", [this]() { return currentPos; });
  logger().addLogEntry("RLController_currentVel", [this]() { return currentVel; });
  logger().addLogEntry("RLController_q_cmd", [this]() { return q_cmd; });
  logger().addLogEntry("RLController_ddot_qp", [this]() { return ddot_qp; });
  logger().addLogEntry("RLController_ddot_qp_w_floatingBase", [this]()
  { return ddot_qp_w_floatingBase; });
  logger().addLogEntry("RLController_tau_qp", [this]() { return tau_qp; });
  logger().addLogEntry("RLController_tau_qp_w_floatingBase", [this]() { return tau_qp_w_floatingBase; });
  logger().addLogEntry("RLController_tau_cmd_after_pd_positionCtl", [this]() { return tau_cmd_after_pd; });

  // RL variables
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
  
  // Controller state variables
  logger().addLogEntry("RLController_useQP", [this]() { return useQP; });
  logger().addLogEntry("RLController_controlledByRL", [this]() { return controlledByRL; });
  logger().addLogEntry("RLController_taskType", [this]() { return taskType; });
}

void RLController::addGui()
{
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::Checkbox("Compensate External Forces", compensateExternalForces));
}

void RLController::initializeRobot(const mc_rtc::Configuration & config)
{
  // H1 joints in mc_rtc/URDF order (based on unitree_sdk2 reorder_obs function)
  mcRtcJointsOrder = {
    "left_hip_yaw_joint",      
    "left_hip_roll_joint",       
    "left_hip_pitch_joint",    
    "left_knee_joint",         
    "left_ankle_joint",        
    "right_hip_yaw_joint",     
    "right_hip_roll_joint",    
    "right_hip_pitch_joint",   
    "right_knee_joint",        
    "right_ankle_joint",       
    "torso_joint",             
    "left_shoulder_pitch_joint",  
    "left_shoulder_roll_joint",     
    "left_shoulder_yaw_joint",    
    "left_elbow_joint",           
    "right_shoulder_pitch_joint", 
    "right_shoulder_roll_joint",  
    "right_shoulder_yaw_joint",   
    "right_elbow_joint"           
  };

  notControlledJoints = {
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "torso_joint"
  };

  dofNumber = robot().mb().nrDof() - 6; // Remove the floating base part (6 DoF)
  refAccel = Eigen::VectorXd::Zero(dofNumber); // TVM
  q_rl_vector = Eigen::VectorXd::Zero(dofNumber);
  q_zero_vector = Eigen::VectorXd::Zero(dofNumber);
  tau_d = Eigen::VectorXd::Zero(dofNumber);
  kp_vector = Eigen::VectorXd::Zero(dofNumber);
  kd_vector = Eigen::VectorXd::Zero(dofNumber);
  high_kp_vector = Eigen::VectorXd::Zero(dofNumber);
  high_kd_vector = Eigen::VectorXd::Zero(dofNumber);
  currentPos = Eigen::VectorXd::Zero(dofNumber);
  currentVel = Eigen::VectorXd::Zero(dofNumber);

  ddot_qp = Eigen::VectorXd::Zero(dofNumber); // Desired acceleration in the QP solver
  ddot_qp_w_floatingBase = Eigen::VectorXd::Zero(robot().mb().nrDof()); // Desired acceleration in the QP solver with floating base
  tau_qp = Eigen::VectorXd::Zero(dofNumber); // Torque in the QP solver
  tau_qp_w_floatingBase = Eigen::VectorXd::Zero(robot().mb().nrDof()); // Torque in the QP solver with floating base
  q_cmd = Eigen::VectorXd::Zero(dofNumber); // The commended position send to the internal PD of the robot
  tau_cmd_after_pd = Eigen::VectorXd::Zero(dofNumber); // The commended position after PD control
  
  // Get the gains from the configuration or set default values
  std::map<std::string, double> kp = config("kp");
  std::map<std::string, double> kd = config("kd");

  std::map<std::string, double> high_kp = config("high_kp");
  std::map<std::string, double> high_kd = config("high_kd");

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
            high_kp_vector[i] = high_kp.at(joint_name);
            high_kd_vector[i] = high_kd.at(joint_name);
            q_rl_vector[i] = t[0];
            q_zero_vector[i] = t[0];
            torque_target[joint_name] = {0.0};
            mc_rtc::log::info("[RLController] Joint {}: currentTargetPosition {}, kp {}, kd {}", joint_name, q_rl_vector[i], kp_vector[i], kd_vector[i]);
            i++;
        }
      }
  }
  current_kp = high_kp_vector;
  current_kd = high_kd_vector;
  solver().removeTask(FSMPostureTask);
}

void RLController::initializeRLPolicy(const mc_rtc::Configuration & config)
{
  auto & real_robot = realRobot(robots()[0].name());

  baseAngVel = real_robot.bodyVelW("pelvis").angular();
  Eigen::Matrix3d baseRot = real_robot.bodyPosW("pelvis").rotation();
  rpy = mc_rbdyn::rpyFromMat(baseRot);
    
  mc_rtc::log::info("[RLController] Posture target initialized with {} joints", dofNumber);

  datastore().make<std::string>("ControlMode", "Torque");
  

  // Initialize reference position and last actions for action blending
  a_before_vector = Eigen::VectorXd::Zero(dofNumber);
  a_vector = Eigen::VectorXd::Zero(dofNumber);
  legPos = Eigen::VectorXd::Zero(10);
  legVel = Eigen::VectorXd::Zero(10);
  legAction = Eigen::VectorXd::Zero(10);

  a_simuOrder = Eigen::VectorXd::Zero(dofNumber);

  mc_rtc::log::info("Reference position initialized with {} joints", q_zero_vector.size());
  q_rl_vector = q_zero_vector;  // Start with reference position
  
  useAsyncInference_ = config("use_async_inference", true);
  mc_rtc::log::info("Async RL inference: {}", useAsyncInference_ ? "enabled" : "disabled");
  currentObservation_ = Eigen::VectorXd::Zero(40);
  currentAction_ = Eigen::VectorXd::Zero(dofNumber);
  latestAction_ = Eigen::VectorXd::Zero(dofNumber);
  
  // Initialize new observation components
  cmd_ = Eigen::Vector3d::Zero();  // Default command (x, y, yaw)
  phase_ = 0.0;  // Phase for periodic gait
  phaseFreq_ = 1.2;  // Phase frequency in Hz
  startPhase_ = std::chrono::steady_clock::now();  // For phase calculation
  
  std::string policyPath = config("policy_path", std::string(""));
  if(!policyPath.empty())
  {
    mc_rtc::log::info("Loading RL policy from: {}", policyPath);
    try {
      rlPolicy_ = std::make_unique<RLPolicyInterface>(policyPath);
      if(rlPolicy_) {
        mc_rtc::log::success("RL policy loaded successfully");
      } else {
        mc_rtc::log::error("RL policy creation failed - policy is null");
      }
    } catch(const std::exception& e) {
      mc_rtc::log::error("Failed to load RL policy: {}", e.what());
      rlPolicy_ = std::make_unique<RLPolicyInterface>(); // Fallback to dummy policy
    }
  }
  else
  {
    mc_rtc::log::warning("No policy_path specified in config, creating dummy policy");
    rlPolicy_ = std::make_unique<RLPolicyInterface>();
  }

  std::string simulator = config("Simulator", std::string(""));
  // check if simulator is Maniskill
  if(simulator == "Maniskill")
  {
    mc_rtc::log::info("Using Maniskill handling");
    policySimulatorHandling_ = std::make_unique<PolicySimulatorHandling>("Maniskill");
  }
  else {
    mc_rtc::log::warning("Simulator not recognized or not set, using default handling");
    policySimulatorHandling_ = std::make_unique<PolicySimulatorHandling>();
  }

  // get list of used joints from config
  std::vector<int> usedJoints = config("Used_joints_index", std::vector<int>{});
  if(!usedJoints.empty())
  {
    std::string jointsStr = "[";
    for(size_t i = 0; i < usedJoints.size(); ++i) {
      if(i > 0) jointsStr += ", ";
      jointsStr += std::to_string(usedJoints[i]);
    }
    jointsStr += "]";
    mc_rtc::log::info("Using custom used joints: {}", jointsStr);
    usedJoints_simuOrder = policySimulatorHandling_->getSimulatorIndices(usedJoints);
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
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> RLController::getPDGains()
{
  std::string robot_name = robot().name();
  std::vector<double> proportionalGains_vec(kp_vector.data(), kp_vector.data() + kp_vector.size());
  std::vector<double> dampingGains_vec(kd_vector.data(), kd_vector.data() + kd_vector.size());
  datastore().call<bool>(robot_name + "::GetPDGains", proportionalGains_vec, dampingGains_vec);
  Eigen::VectorXd p_vec = Eigen::VectorXd::Map(proportionalGains_vec.data(), proportionalGains_vec.size());
  Eigen::VectorXd d_vec = Eigen::VectorXd::Map(dampingGains_vec.data(), dampingGains_vec.size());
  mc_rtc::log::info("[RLController] Current PD Gains for {} are:\n\tkp = {}\n\tkd = {}", robot_name, p_vec.transpose(), d_vec.transpose());
  return std::make_tuple(p_vec, d_vec);
}

bool RLController::setPDGains(Eigen::VectorXd p_vec, Eigen::VectorXd d_vec)
{
  std::string robot_name = robot().name();
  // Update kp and kd use by the controller
  current_kp = p_vec;
  current_kd = d_vec;

  // Update kp and kd use by the robot or simulator (Internal PD)
  mc_rtc::log::info("[RLController] Setting PD gains for {}:\n\tkp = {}\n\tkd = {}", robot_name, p_vec.transpose(), d_vec.transpose());
  const std::vector<double> proportionalGains_vec(p_vec.data(), p_vec.data() + p_vec.size());
  const std::vector<double> dampingGains_vec(d_vec.data(), d_vec.data() + d_vec.size());
  return datastore().call<bool>(robot_name + "::SetPDGains", proportionalGains_vec, dampingGains_vec);
}

bool RLController::isHighGain(double tol)
{
  // Update kp and kd use by the controller
  std::tie(current_kp, current_kd) = getPDGains();
  // Check if the current gains are close to the high gains
  bool highGain = ((current_kp - high_kp_vector).norm() < tol) && ((current_kd - high_kd_vector).norm() < tol);
  mc_rtc::log::info("[RLController] isHighGain: {}", highGain);
  return highGain;
}

void RLController::initializeState(bool torque_control, int task_type, bool controlled_by_rl)
{
  if(torque_control) datastore().get<std::string>("ControlMode") = "Torque";
  else datastore().get<std::string>("ControlMode") = "Position";
  useQP = true;
  if(task_type == PURE_RL) useQP = false;
  else taskType = task_type;
  controlledByRL = controlled_by_rl;
  if(controlledByRL)
  {
    // Set low gains for RL
    if(isHighGain()) setPDGains(kp_vector, kd_vector);
    tasksComputation(q_rl_vector);
  }
  else
  {
    // Set high gains for model-based control
    if(!isHighGain()) setPDGains(high_kp_vector, high_kd_vector);
    tasksComputation(q_zero_vector);
  }
}
