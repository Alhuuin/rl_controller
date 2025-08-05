#include "RLController.h"
#include <Eigen/src/Core/VectorBlock.h>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <mc_rtc/logging.h>
#include <mc_rbdyn/configuration_io.h>
#include <chrono>
#include <fstream>
#include <sstream>

RLController::RLController(mc_rbdyn::RobotModulePtr rm, double dt, 
                           const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(rm, dt, config, Backend::TVM)
{
  // Add constraints
  // contactConstraintTest =std::make_unique<mc_solver::ContactConstraint>(timeStep, mc_solver::ContactConstraint::ContactType::Acceleration);
  // solver().addConstraintSet(contactConstraintTest);

  // selfCollisionConstraint->setCollisionsDampers(solver(), {1.8, 70.0});`
  logTiming_ = config("log_timing");
  timingLogInterval_ = config("timing_log_interval");
  
  dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(
      new mc_solver::DynamicsConstraint(
          robots(), 0, {0.1, 0.01, 0.0, 1.8, 70.0}, 0.9, true));
  solver().addConstraintSet(dynamicsConstraint);
  
  dofNumber_with_floatingBase = robot().mb().nrDof();
  dofNumber = robot().mb().nrDof() - 6; // Remove the floating base part (6 DoF)
  // refAccel = Eigen::VectorXd::Zero(dofNumber_with_floatingBase); // Task
  refAccel = Eigen::VectorXd::Zero(dofNumber); // TVM
  // refAccel_w_floatingBase = Eigen::VectorXd::Zero(dofNumber_with_floatingBase);
  q_rl_vector = Eigen::VectorXd::Zero(dofNumber);
  tau_d = Eigen::VectorXd::Zero(dofNumber);
  // tau_d_w_floatingBase = Eigen::VectorXd::Zero(dofNumber_with_floatingBase);
  kp_vector = Eigen::VectorXd::Zero(dofNumber);
  kd_vector = Eigen::VectorXd::Zero(dofNumber);
  high_kp_vector = Eigen::VectorXd::Zero(dofNumber);
  high_kd_vector = Eigen::VectorXd::Zero(dofNumber);
  currentPos = Eigen::VectorXd::Zero(dofNumber);
  currentVel = Eigen::VectorXd::Zero(dofNumber);
  // currentPos_w_floatingBase = Eigen::VectorXd::Zero(dofNumber_with_floatingBase);
  // currentVel_w_floatingBase = Eigen::VectorXd::Zero(dofNumber_with_floatingBase);

  ddot_qp = Eigen::VectorXd::Zero(dofNumber); // Desired acceleration in the QP solver
  ddot_qp_w_floatingBase = Eigen::VectorXd::Zero(dofNumber_with_floatingBase); // Desired acceleration in the QP solver with floating base
  q_cmd = Eigen::VectorXd::Zero(dofNumber); // The commended position send to the internal PD of the robot
  // q_cmd_w_floatingBase = Eigen::VectorXd::Zero(dofNumber_with_floatingBase); // The commended position send to the internal PD of the robot with floating base
  tau_cmd_after_pd = Eigen::VectorXd::Zero(dofNumber); // The commended position after PD control
  
  similiTorqueTask = std::make_shared<mc_tasks::PostureTask>(solver(), robot().robotIndex(), 0.0, 1000.0);
  similiTorqueTask->weight(1000.0);
  similiTorqueTask->stiffness(0.0);
  similiTorqueTask->damping(0.0);
  similiTorqueTask->refAccel(refAccel);
  solver().addTask(similiTorqueTask);

  // Get the gains from the configuration or set default values
  kp = config("kp");
  kd = config("kd");

  high_kp = config("high_kp");
  high_kd = config("high_kd");

  // Get the default posture target from the robot's posture task
  FSMPostureTask = getPostureTask(robot().name());
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
            mc_rtc::log::info("[RLController] Joint {}: currentTargetPosition {}, kp {}, kd {}", joint_name, q_rl_vector[i], kp_vector[i], kd_vector[i]);
            i++;
        }
      }
  }
  solver().removeTask(FSMPostureTask);

  auto & real_robot = realRobot(robots()[0].name());
  
  //  Eigen::Vector3d baseAngVel = real_robot.bodyVelW()[0].angular();

  baseAngVel = real_robot.bodyVelW("pelvis").angular();
  Eigen::Matrix3d baseRot = real_robot.bodyPosW("pelvis").rotation();
  rpy = mc_rbdyn::rpyFromMat(baseRot);
    
  mc_rtc::log::info("[RLController] Posture target initialized with {} joints", dofNumber);

  datastore().make<std::string>("ControlMode", "Torque");
  // datastore().make<std::string>("ControlMode", "Position");
  initializeAllJoints();
  // initializeImpedanceGains();
  a_maniskillOrder = Eigen::VectorXd::Zero(dofNumber);
  
  // Initialize reference position and last actions for action blending
  q_zero_vector = Eigen::VectorXd::Zero(dofNumber);
  a_before_vector = Eigen::VectorXd::Zero(dofNumber);
  a_vector = Eigen::VectorXd::Zero(dofNumber);
  legPos = Eigen::VectorXd::Zero(10);
  legVel = Eigen::VectorXd::Zero(10);
  legAction = Eigen::VectorXd::Zero(10);

  // Capture current joint positions as reference (default position)
  const auto & robot = this->robot();
  for(size_t i = 0; i < mcRtcJointsOrder.size(); ++i)
  {
    if(robot.hasJoint(mcRtcJointsOrder[i]))
    {
      auto jIndex = robot.jointIndexByName(mcRtcJointsOrder[i]);
      q_zero_vector(i) = robot.mbc().q[jIndex][0];
    }
    else
    {
      mc_rtc::log::warning("Joint {} not found in robot model during reference initialization", mcRtcJointsOrder[i]);
      q_zero_vector(i) = 0.0;
    }
  }
  
  mc_rtc::log::info("Reference position initialized with {} joints", q_zero_vector.size());
  // mc_rtc::log::info("Reference positions: {}", q_ref_.transpose().format(Eigen::IOFormat(3, 0, ", ", "", "", "", "[", "]")));
  
  // Initialize 40Hz inference timing
  lastInferenceTime_ = std::chrono::steady_clock::now();
  q_rl_vector = q_zero_vector;  // Start with reference position
  targetPositionValid_ = true;
  
  useAsyncInference_ = config("use_async_inference", true);
  mc_rtc::log::info("Async RL inference: {}", useAsyncInference_ ? "enabled" : "disabled");
  
  shouldStopInference_ = false;
  newObservationAvailable_ = false;
  newActionAvailable_ = false;
  inferenceCounter_ = 0;  // Initialize inference counter
  currentObservation_ = Eigen::VectorXd::Zero(35);
  currentAction_ = Eigen::VectorXd::Zero(dofNumber);
  latestAction_ = Eigen::VectorXd::Zero(dofNumber);
  
  std::string policyPath = config("policy_path", std::string(""));
  if(!policyPath.empty())
  {
    mc_rtc::log::info("Loading RL policy from: {}", policyPath);
    rlPolicy_ = std::make_unique<RLPolicyInterface>(policyPath);
  }
  else
  {
    mc_rtc::log::warning("No policy_path specified, creating dummy policy");
    rlPolicy_ = std::make_unique<RLPolicyInterface>();
  }
  
  // torqueTask_ = std::make_shared<mc_tasks::TorqueTask>(solver(), 0, 1000.0);
  // solver().addTask(torqueTask_);
  
  if(useAsyncInference_)
  {
    startInferenceThread();
  }

  logging();

  // Load CSV input data during initialization
  loadCSVInputData();

  mc_rtc::log::warning(csvInputData_[160]);
  

  mc_rtc::log::success("RLController init");
}

void RLController::logging()
{
  logger().addLogEntry("RLController_refAccel", [this]() { return refAccel; });
  // logger().addLogEntry("RLController_refAccel_w_floatingBase", [this]()
  // { return refAccel_w_floatingBase; });
  logger().addLogEntry("RLController_currentTargetPosition", [this]() { return q_rl_vector; });
  logger().addLogEntry("RLController_tau_d", [this]() { return tau_d; });
  // logger().addLogEntry("RLController_tau_d_w_floatingBase", [this]()
  // { return tau_d_w_floatingBase; });
  logger().addLogEntry("RLController_kp", [this]() { return kp_vector; });
  logger().addLogEntry("RLController_kd", [this]() { return kd_vector; });
  logger().addLogEntry("RLController_currentPos", [this]() { return currentPos; });
  // logger().addLogEntry("RLController_currentPos_w_floatingBase", [this]()
  // { return currentPos_w_floatingBase; });
  logger().addLogEntry("RLController_currentVel", [this]() { return currentVel; });
  // logger().addLogEntry("RLController_currentVel_w_floatingBase", [this]()
  // { return currentVel_w_floatingBase; });
  logger().addLogEntry("RLController_q_cmd", [this]() { return q_cmd; });
  // logger().addLogEntry("RLController_q_cmd_w_floatingBase", [this]()
  // { return q_cmd_w_floatingBase; });
  logger().addLogEntry("RLController_ddot_qp", [this]() { return ddot_qp; });
  logger().addLogEntry("RLController_ddot_qp_w_floatingBase", [this]()
  { return ddot_qp_w_floatingBase; });

  logger().addLogEntry("RLController_tau_cmd_after_pd_positionCtl", [this]() { return tau_cmd_after_pd; });

  // pastAction_
  logger().addLogEntry("RLController_pastAction", [this]() { return a_maniskillOrder; });
  // q_ref_
  logger().addLogEntry("RLController_qZero", [this]() { return q_zero_vector; });
  // lastActions_
  logger().addLogEntry("RLController_a_before", [this]() { return a_before_vector; });
  // currentObservation_
  logger().addLogEntry("RLController_currentObservation", [this]() { return currentObservation_; });
  // a_vector
  logger().addLogEntry("RLController_a_vector", [this]() { return a_vector; });
  // a_maniskillOrder
  logger().addLogEntry("RLController_a_maniskillOrder", [this]() { return a_maniskillOrder; });
  // currentAction_
  logger().addLogEntry("RLController_currentAction", [this]() { return currentAction_; });
  // latestAction_
  logger().addLogEntry("RLController_latestAction", [this]() { return latestAction_; });
  // baseAngVel
  logger().addLogEntry("RLController_baseAngVel", [this]() { return baseAngVel; });
  // rpy
  logger().addLogEntry("RLController_rpy", [this]() { return rpy; });
  // legPos
  logger().addLogEntry("RLController_legPos", [this]() { return legPos; });
  // legVel
  logger().addLogEntry("RLController_legVel", [this]() { return legVel; });
  // legAction
  logger().addLogEntry("RLController_legAction", [this]() { return legAction; });
}

RLController::~RLController()
{
  stopInferenceThread();
  mc_rtc::log::info("RLController destroyed");
}

bool RLController::run()
{
  // Update the solver depending on the control mode
  auto ctrl_mode = datastore().get<std::string>("ControlMode");
  if (static_pos)
  {
    auto run = mc_control::fsm::Controller::run(mc_solver::FeedbackType::ClosedLoopIntegrateReal);
    robot().forwardKinematics();
    robot().forwardVelocity();
    robot().forwardAcceleration();


    Eigen::MatrixXd Kp_inv = kp_vector.cwiseInverse().asDiagonal();

    q_cmd = currentPos + Kp_inv*(high_kp_vector*(q_zero_vector - currentPos) - currentVel.cwiseProduct(high_kd_vector - kd_vector));

    if (ctrl_mode.compare("Position") == 0)
    {
      auto q = robot().mbc().q;
      
      size_t i = 0;
      for (const auto &joint_name : jointNames)
      {
        q[robot().jointIndexByName(joint_name)][0] = q_cmd[i];
        i++;
      }

      robot().mbc().q = q; // Update the mbc with the new position
    }
    else
    { 
      tau_cmd_after_pd = kd_vector.cwiseProduct(currentPos - q_cmd) - kd_vector.cwiseProduct(currentVel); // PD control to get the commanded position after PD control
      auto tau = robot().mbc().jointTorque;

      size_t i = 0;
      for (const auto &joint_name : jointNames)
      {
        tau[robot().jointIndexByName(joint_name)][0] = tau_cmd_after_pd[i];
        i++;
      }

      robot().mbc().jointTorque = tau; // Update the mbc with the new position
    }
    return run;
  }
  if (ctrl_mode.compare("Position") == 0) {
    auto run = mc_control::fsm::Controller::run(mc_solver::FeedbackType::ClosedLoopIntegrateReal);
    robot().forwardKinematics();
    robot().forwardVelocity();
    robot().forwardAcceleration();
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

    Eigen::MatrixXd Kp_inv = kp_vector.cwiseInverse().asDiagonal();

    q_cmd = currentPos + Kp_inv*(M*ddot_qp + Cg + kd_vector.cwiseProduct(currentVel)); // Inverse PD control to get the commanded position <=> RL position control

    tau_cmd_after_pd = kd_vector.cwiseProduct(currentPos - q_cmd) - kd_vector.cwiseProduct(currentVel); // PD control to get the commanded position after PD control
    
    auto q = robot().mbc().q;
    
    size_t i = 0;
    for (const auto &joint_name : jointNames)
    {
      q[robot().jointIndexByName(joint_name)][0] = q_cmd[i];
      i++;
    }

    robot().mbc().q = q; // Update the mbc with the new position

    if (useQP)
      return run;
    else
    {
      auto q = robot().encoderValues();
      currentPos = Eigen::VectorXd::Map(q.data(), q.size());
      auto vel = robot().encoderVelocities();
      currentVel = Eigen::VectorXd::Map(vel.data(), vel.size());
      auto tau = robot().mbc().jointTorque;
      tau_d = kp_vector.cwiseProduct(q_zero_vector - currentPos) + kd_vector.cwiseProduct(-currentVel);
      
      size_t i = 0;
      for (const auto &joint_name : jointNames)
      {
        tau[robot().jointIndexByName(joint_name)][0] = tau_d[i];
        i++;
      }

      robot().mbc().jointTorque = tau; // Update the mbc with the new position
      return true;
    }

  } 
  if (useQP == false)
  {
    mc_control::fsm::Controller::run(mc_solver::FeedbackType::ClosedLoopIntegrateReal);
    robot().forwardKinematics();
    robot().forwardVelocity();
    robot().forwardAcceleration();
    auto q = robot().encoderValues();
    currentPos = Eigen::VectorXd::Map(q.data(), q.size());
    auto vel = robot().encoderVelocities();
    currentVel = Eigen::VectorXd::Map(vel.data(), vel.size());
    auto tau = robot().mbc().jointTorque;
    tau_d = kp_vector.cwiseProduct(q_zero_vector - currentPos) + kd_vector.cwiseProduct(-currentVel);
    
    size_t i = 0;
    for (const auto &joint_name : jointNames)
    {
      tau[robot().jointIndexByName(joint_name)][0] = tau_d[i];
      i++;
    }

    robot().mbc().jointTorque = tau; // Update the mbc with the new position
    return true;
  }
  return mc_control::fsm::Controller::run(
      mc_solver::FeedbackType::ClosedLoopIntegrateReal);

}

void RLController::reset(const mc_control::ControllerResetData & reset_data)
{
  mc_control::fsm::Controller::reset(reset_data);
  mc_rtc::log::success("RLController reset completed");
}

void RLController::initializeAllJoints()
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

  maniskillJointsOrder = {
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "torso_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_joint",
    "right_ankle_joint",
    "left_elbow_joint",
    "right_elbow_joint"
  };

  maniskillLegsJointsOrder = {
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_joint",
    "right_ankle_joint"
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
  
  // leg joints (first 10)
  legJoints_ = std::vector<std::string>(mcRtcJointsOrder.begin(), mcRtcJointsOrder.begin() + 10);
  
  legIndicesInManiskill = {0, 1, 3, 4, 7, 8, 11, 12, 15, 16};
  maniskillToMcRtcIdx_ = {0, 3, 7, 11, 15, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 6, 10, 14, 18};
  mcRtcToManiskillIdx_ = {0, 5, 10, 1, 6, 11, 15, 2, 7, 12, 16, 3, 8, 13, 17, 4, 9, 14, 18};
}

Eigen::VectorXd RLController::reorderJointsToManiskill(const Eigen::VectorXd & obs)
{  
  if(obs.size() != dofNumber) {
    mc_rtc::log::error("Observation reordering expects dofNumber joints, got {}", obs.size());
    return obs;
  }
  
  Eigen::VectorXd reordered(dofNumber);
  for(int i = 0; i < dofNumber; i++) {
    if(i >= mcRtcToManiskillIdx_.size()) {
      mc_rtc::log::error("Trying to access mcRtcToManiskillIdx_[{}] but size is {}", i, mcRtcToManiskillIdx_.size());
      reordered[i] = 0.0;
      continue;
    }
    
    int srcIdx = mcRtcToManiskillIdx_[i];
    if(srcIdx >= obs.size()) {
      mc_rtc::log::error("Index {} out of bounds for obs size {}", srcIdx, obs.size());
      reordered[i] = 0.0;
    } else {
      reordered[i] = obs(srcIdx);
    }
  }
  return reordered;
}

Eigen::VectorXd RLController::reorderJointsFromManiskill(const Eigen::VectorXd & action)
{  
  if(action.size() != dofNumber) {
    mc_rtc::log::error("Action reordering expects dofNumber joints, got {}", action.size());
    return action;
  }
  
  Eigen::VectorXd reordered(dofNumber);
  for(int i = 0; i < dofNumber; i++) {
    if(i >= maniskillToMcRtcIdx_.size()) {
      mc_rtc::log::error("Trying to access maniskillToMcRtcIdx_[{}] but size is {}", i, maniskillToMcRtcIdx_.size());
      reordered[i] = 0.0;
      continue;
    }
    
    int srcIdx = maniskillToMcRtcIdx_[i];
    if(srcIdx >= action.size()) {
      mc_rtc::log::error("Action reorder index {} out of bounds for action size {}", srcIdx, action.size());
      reordered[i] = 0.0;
    } else {
      reordered[i] = action(srcIdx);
    }
  }
  return reordered;
}

Eigen::VectorXd RLController::getCurrentObservation()
{
  // Observation: [base angular velocity (3), roll (1), pitch (1), joint pos (10), joint vel (10), past action (10)]
  
  Eigen::VectorXd obs(35);
  obs = Eigen::VectorXd::Zero(35);
  
  // const auto & robot = this->robot();

  auto & robot = robots()[0];
  auto & real_robot = realRobot(robots()[0].name());
  
  //  Eigen::Vector3d baseAngVel = real_robot.bodyVelW()[0].angular();

  // baseAngVel = real_robot.bodyVelW("pelvis").angular();
  baseAngVel = real_robot.bodyVelW("pelvis").angular();
  obs.segment(0, 3) = baseAngVel; //base angular vel
  
  Eigen::Matrix3d baseRot = real_robot.bodyPosW("pelvis").rotation();
  // Eigen::Matrix3d baseRot = real_robot.bodyTransform("pelvis").rotation();
  rpy = mc_rbdyn::rpyFromMat(baseRot);
  obs(3) = rpy(0);  // roll
  obs(4) = rpy(1);  // pitch
  
  Eigen::VectorXd reorderedPos = reorderJointsToManiskill(currentPos);
  Eigen::VectorXd reorderedVel = reorderJointsToManiskill(currentVel);
  
  for(size_t i = 0; i < legIndicesInManiskill.size(); ++i)
  {
    int idx = legIndicesInManiskill[i];
    if(idx >= reorderedPos.size()) {
      mc_rtc::log::error("Leg joint index {} out of bounds for reordered size {}", idx, reorderedPos.size());
      legPos(i) = 0.0;
      legVel(i) = 0.0;
    } else {
      legPos(i) = reorderedPos(idx);
      legVel(i) = reorderedVel(idx);
    }
  }
  
  obs.segment(5, 10) = legPos;
  obs.segment(15, 10) = legVel;
  
  // past action: reorder to ManiSkill format and extract leg joints
  for(size_t i = 0; i < legIndicesInManiskill.size(); ++i)
  {
    int idx = legIndicesInManiskill[i];
    if(idx >= a_maniskillOrder.size()) {
      mc_rtc::log::error("Past action index {} out of bounds for size {}", idx, a_maniskillOrder.size());
      legAction(i) = 0.0;
    } else {
      legAction(i) = a_maniskillOrder(idx);
    }
  }
  obs.segment(25, 10) = legAction;

  // Get the observation from CSV using thread-safe counter
  if(!csvInputData_.empty()) {
    int currentCounter = inferenceCounter_.load();
    obs = csvInputData_[currentCounter % 297];
  }

  return obs;
}

void RLController::applyAction(const Eigen::VectorXd & action)
{
  if(action.size() != dofNumber)
  {
    mc_rtc::log::error("Action size mismatch: expected dofNumber, got {}", action.size());
    return;
  }
  
  // Check if it's time for new inference (40Hz = 25ms period)
  auto currentTime = std::chrono::steady_clock::now();
  auto timeSinceLastInference = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastInferenceTime_);
  
  bool shouldRunInference = timeSinceLastInference.count() >= INFERENCE_PERIOD_MS;
  
  if(shouldRunInference) {
    mc_rtc::log::warning(shouldRunInference);
    
    // Get current observation for logging
    Eigen::VectorXd currentObs = getCurrentObservation();
    
    // Run new inference and update target position
    a_vector = reorderJointsFromManiskill(action);
    
    // Apply action blending formula: target_qpos = default_qpos + 0.75 * action + 0.25 * previous_actions
    q_rl_vector = q_zero_vector + 0.75 * a_vector + 0.25 * a_before_vector;

    // For not controlled joints, use the zero position
    for(const auto & joint : notControlledJoints)
    {
      auto it = std::find(mcRtcJointsOrder.begin(), mcRtcJointsOrder.end(), joint);
      if(it != mcRtcJointsOrder.end())
      {
        size_t idx = std::distance(mcRtcJointsOrder.begin(), it);
        if(idx < q_rl_vector.size())
        {
          q_rl_vector(idx) = q_zero_vector(idx); // Set to zero position
        }
        else
        {
          mc_rtc::log::error("Joint {} index {} out of bounds for q_rl_vector size {}", joint, idx, q_rl_vector.size());
        }
      }
      else
      {
        mc_rtc::log::error("Joint {} not found in mcRtcJointsOrder", joint);
      }
    }
    
    // Update lastActions_ for next iteration
    a_before_vector = a_vector;
    a_maniskillOrder = reorderJointsToManiskill(a_vector);
    
    // Update timing
    lastInferenceTime_ = currentTime;
    targetPositionValid_ = true;

    
    // if(inferenceCounter % 10 == 0) { // Log every 10 inferences (0.25 seconds at 40Hz)
    mc_rtc::log::info("=== RLController Policy I/O Inference #{} === ", inferenceCounter_.load());
    mc_rtc::log::info("Policy Input (35 obs): [");
    for(int i = 0; i < 35; ++i) {
      mc_rtc::log::info("  [{}]: {:.6f}", i, currentObs(i));
    }
    mc_rtc::log::info("]");
    mc_rtc::log::info("Policy Output Raw (dofNumber ManiSkill order): [");
    for(int i = 0; i < dofNumber; ++i) {
      mc_rtc::log::info("  [{}]: {:.6f}", i, action(i));
    }
    mc_rtc::log::info("]");
    mc_rtc::log::info("Policy Output Reordered (dofNumber mc_rtc order): [");
    for(int i = 0; i < dofNumber; ++i) {
      mc_rtc::log::info("  [{}]: {:.6f}", i, a_vector(i));
    }
    mc_rtc::log::info("]");
    mc_rtc::log::info("Blended Target Position (dofNumber): [");
    for(int i = 0; i < dofNumber; ++i) {
      mc_rtc::log::info("  [{}]: {:.6f}", i, q_rl_vector(i));
    }
    mc_rtc::log::info("]");
    mc_rtc::log::info("=== End Policy I/O ===");


    inferenceCounter_++;
  }
  
  // Always apply impedance control at mc_rtc frequency using current target position
  if(!targetPositionValid_) {
    mc_rtc::log::warning("No valid target position available for impedance control");
    return;
  }
  
  // Get current joint positions and velocities
  Eigen::VectorXd q_current(dofNumber);
  Eigen::VectorXd q_dot_current(dofNumber);
  auto & real_robot = realRobot(robots()[0].name());
  auto q = real_robot.encoderValues();
  q_current = Eigen::VectorXd::Map(q.data(), q.size());
  // currentPos_w_floatingBase = Eigen::VectorXd::Map(q.data(), q.size());
  // currentPos = currentPos_w_floatingBase.head(dofNumber); // Exclude the floating base part
  auto vel = real_robot.encoderVelocities();
  q_dot_current = Eigen::VectorXd::Map(vel.data(), vel.size());

  const auto & robot = this->robot();
  
  for(size_t i = 0; i < mcRtcJointsOrder.size(); ++i)
  {
    if(robot.hasJoint(mcRtcJointsOrder[i]))
    {
      auto jIndex = robot.jointIndexByName(mcRtcJointsOrder[i]);
      q_current(i) = robot.mbc().q[jIndex][0];
      q_dot_current(i) = robot.mbc().alpha[jIndex][0];
    }
    else
    {
      q_current(i) = 0.0;
      q_dot_current(i) = 0.0;
    }
  }
  torqueTaskSimulation(q_rl_vector);
}

void RLController::startInferenceThread()
{
  mc_rtc::log::info("Starting RL inference thread");
  inferenceThread_ = std::make_unique<std::thread>(&RLController::inferenceThreadFunction, this);
}

void RLController::stopInferenceThread()
{
  if(inferenceThread_ && inferenceThread_->joinable())
  {
    mc_rtc::log::info("Stopping RL inference thread");
    shouldStopInference_ = true;
    inferenceCondition_.notify_one();
    inferenceThread_->join();
    inferenceThread_.reset();
  }
}

void RLController::inferenceThreadFunction()
{
  mc_rtc::log::info("RL inference thread started");
  
  while(!shouldStopInference_)
  {
    //wait for new observation or stop signal
    std::unique_lock<std::mutex> lock(observationMutex_);
    inferenceCondition_.wait(lock, [this] { 
      return newObservationAvailable_.load() || shouldStopInference_.load(); 
    });
    
    if(shouldStopInference_) break;
    
    // copy observation for processing
    Eigen::VectorXd obs = currentObservation_;
    newObservationAvailable_ = false;
    lock.unlock();
    
    try
    {
      auto startTime = std::chrono::high_resolution_clock::now();
      Eigen::VectorXd action = rlPolicy_->predict(obs);
      auto endTime = std::chrono::high_resolution_clock::now();
      
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
      
      //update shared action
      {
        std::lock_guard<std::mutex> actionLock(actionMutex_);
        currentAction_ = action;
        newActionAvailable_ = true;
      }
    }
    catch(const std::exception & e)
    {
      mc_rtc::log::error("RL inference error: {}", e.what());
      //keep using the previous action
    }
  }
  
  mc_rtc::log::info("RL inference thread stopped");
}

void RLController::updateObservationForInference()
{ 
  Eigen::VectorXd obs = getCurrentObservation();
  // Get CSV input data for this inference step using thread-safe counter
  Eigen::VectorXd csvInput;
  if(!csvInputData_.empty()) {
    int currentCounter = inferenceCounter_.load();
    csvInput = csvInputData_[currentCounter % 297];

    // Log CSV input if available
    if(csvInput.size() > 0) {
      mc_rtc::log::info("CSV Input Array (size {}): [", csvInput.size());
      for(int i = 0; i < csvInput.size(); ++i) {
        mc_rtc::log::info("  [{}]: {:.6f}", i, csvInput(i));
      }
      mc_rtc::log::info("]");
    }

    obs = csvInput;
  }
  
  //update shared observation
  {
    std::lock_guard<std::mutex> lock(observationMutex_);
    currentObservation_ = obs;
    newObservationAvailable_ = true;
  }
  
  // notify inference thread
  inferenceCondition_.notify_one();
}

Eigen::VectorXd RLController::getLatestAction()
{
  if(newActionAvailable_)
  {
    std::lock_guard<std::mutex> lock(actionMutex_);
    if(newActionAvailable_)
    {
      latestAction_ = currentAction_;
      newActionAvailable_ = false;
    }
  }
  return latestAction_;
} 

void RLController::torqueTaskSimulation(Eigen::VectorXd & currentTargetPosition)
{
  auto & robot = robots()[0];
  auto & real_robot = realRobot(robots()[0].name());

  auto q = real_robot.encoderValues();
  currentPos = Eigen::VectorXd::Map(q.data(), q.size());
  // currentPos_w_floatingBase = Eigen::VectorXd::Map(q.data(), q.size());
  // currentPos = currentPos_w_floatingBase.head(dofNumber); // Exclude the floating base part
  auto vel = real_robot.encoderVelocities();
  currentVel = Eigen::VectorXd::Map(vel.data(), vel.size());
  // currentVel_w_floatingBase = Eigen::VectorXd::Map(vel.data(), vel.size());
  // currentVel = currentVel_w_floatingBase.head(dofNumber); // Exclude the floating base part

  // mc_rtc::log::info("[RLController] Current Position: {}", currentPos);
  // mc_rtc::log::info("[RLController] Current Velocity: {}", currentVel);

  tau_d = kp_vector.cwiseProduct(currentTargetPosition - currentPos) + kd_vector.cwiseProduct(-currentVel);
  // mc_rtc::log::info("[RLController] tau_d: {}", tau_d);

  // tau_d_w_floatingBase = Eigen::VectorXd::Zero(dofNumber_with_floatingBase); // full vector zeroed
  // tau_d_w_floatingBase.head(dofNumber) = tau_d; // copy only joint torques


  // Simulate the torque task by converting the torque target to an acceleration target
  rbd::ForwardDynamics fd(real_robot.mb());
  fd.computeH(real_robot.mb(), real_robot.mbc());
  fd.computeC(real_robot.mb(), real_robot.mbc());
  Eigen::MatrixXd M_w_floatingBase = fd.H();
  Eigen::VectorXd Cg_w_floatingBase = fd.C();
  Eigen::MatrixXd M = M_w_floatingBase.bottomRightCorner(dofNumber, dofNumber);
  Eigen::VectorXd Cg = Cg_w_floatingBase.tail(dofNumber);
  auto extTorqueSensor = robot.device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
  Eigen::VectorXd externalTorques = Eigen::VectorXd::Zero(dofNumber);
  Eigen::VectorXd torques = extTorqueSensor.torques();
  if (torques.hasNaN()) {
    mc_rtc::log::error("[RLController] External torques contain NaN values, using zero torques instead");
  } else {
    externalTorques = torques;
  }
  // refAccel = M.completeOrthogonalDecomposition().solve(tau_d - Cg + externalTorques);
  refAccel = M.completeOrthogonalDecomposition().solve(tau_d - Cg);
  //Choleesky decomposition solveinplace version:
  // refAccel = M.llt().solve(tau_d - Cg); // This is the acceleration target for the QP solver

  // For the TVM backend
  // refAccel = Eigen::VectorXd::Zero(dofNumber_with_floatingBase);
  // refAccel.head(dofNumber) = refAccel_w_floatingBase.head(dofNumber);
  // Keep the full vector for Task
  // refAccel = refAccel_w_floatingBase; 


  // mc_rtc::log::info("[RLController] refAccel: {}", refAccel);
  similiTorqueTask->refAccel(refAccel);
}

void RLController::loadCSVInputData()
{
  std::ifstream file("eval_policy_io.csv");
  if(!file.is_open()) {
    mc_rtc::log::warning("Cannot open CSV file 'eval_policy_io.csv', CSV input data will be empty");
    return;
  }
  
  std::string line;
  std::vector<std::string> headers;
  int input_col = -1;

  // Read header
  if(std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string col;
    while(std::getline(ss, col, ','))
    {
      // Trim whitespace from column header
      col.erase(0, col.find_first_not_of(" \t\r\n"));
      col.erase(col.find_last_not_of(" \t\r\n") + 1);
      headers.push_back(col);
      if(col == "input") input_col = headers.size() - 1;
    }
    
    if(input_col == -1) {
      mc_rtc::log::warning("Column 'input' not found in CSV file, CSV input data will be empty");
      return;
    }
  }
  else
  {
    mc_rtc::log::warning("Cannot read header from CSV file, CSV input data will be empty");
    return;
  }

  // Parse all data lines
  int line_number = 1; // Start from line 1 (after header)
  while(std::getline(file, line))
  {
    // Parse CSV line properly handling quoted fields
    std::vector<std::string> csv_cells;
    std::string current_cell;
    bool in_quotes = false;
    
    for(size_t i = 0; i < line.length(); ++i) {
      char c = line[i];
      if(c == '"') {
        in_quotes = !in_quotes;
      } else if(c == ',' && !in_quotes) {
        csv_cells.push_back(current_cell);
        current_cell.clear();
      } else {
        current_cell += c;
      }
    }
    csv_cells.push_back(current_cell); // Add the last cell
    
    // Parse the input column
    if(input_col < csv_cells.size()) {
      std::string input_str = csv_cells[input_col];
      
      // Remove surrounding quotes if present
      if(!input_str.empty() && input_str.front() == '"') {
        input_str = input_str.substr(1);
      }
      if(!input_str.empty() && input_str.back() == '"') {
        input_str.pop_back();
      }
      
      std::vector<double> temp_values;
      if(!input_str.empty()) {
        // Parse entries like "[0]: 0.139669, [1]: -0.032801, ..."
        std::stringstream value_ss(input_str);
        std::string token;
        
        while(std::getline(value_ss, token, ',')) {
          // Trim whitespace
          token.erase(0, token.find_first_not_of(" \t"));
          token.erase(token.find_last_not_of(" \t") + 1);
          
          // Find the colon separator
          size_t colon_pos = token.find(':');
          if(colon_pos != std::string::npos && colon_pos + 1 < token.length()) {
            // Extract the value after the colon
            std::string value_str = token.substr(colon_pos + 1);
            
            // Trim whitespace from the value
            value_str.erase(0, value_str.find_first_not_of(" \t"));
            value_str.erase(value_str.find_last_not_of(" \t") + 1);
            
            if(!value_str.empty()) {
              try {
                double parsed_value = std::stod(value_str);
                temp_values.push_back(parsed_value);
              } catch(const std::exception&) {
                // Ignore parsing errors silently
              }
            }
          }
        }
      }
      
      // Convert to Eigen::VectorXd and store
      if(!temp_values.empty()) {
        Eigen::VectorXd input_vector = Eigen::Map<Eigen::VectorXd>(temp_values.data(), temp_values.size());
        csvInputData_.push_back(input_vector);
      } else {
        // Add empty vector for this line
        csvInputData_.push_back(Eigen::VectorXd::Zero(0));
      }
    } else {
      // Add empty vector if column not found
      csvInputData_.push_back(Eigen::VectorXd::Zero(0));
    }
    
    line_number++;
  }
  
  mc_rtc::log::info("Loaded {} lines of CSV input data", csvInputData_.size());
  if(!csvInputData_.empty() && csvInputData_[0].size() > 0) {
    mc_rtc::log::info("First input vector has {} elements", csvInputData_[0].size());
  }
}

