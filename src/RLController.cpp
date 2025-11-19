#include "RLController.h"
#include <Eigen/src/Core/VectorBlock.h>
#include <RBDyn/Jacobian.h>
#include <RBDyn/MultiBodyConfig.h>
#include <SpaceVecAlg/SpaceVecAlg>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <mc_rtc/gui/ArrayInput.h>
#include <mc_rtc/gui/NumberInput.h>
#include <mc_rtc/logging.h>
#include <mc_rbdyn/configuration_io.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <utility>


RLController::RLController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(rm, dt, config, Backend::TVM)
{
  logTiming_ = config("log_timing");
  timingLogInterval_ = config("timing_log_interval");
  isWalkingPolicy = config("is_walking_policy", false);
  isPerfectPolicy = config("is_perfect_policy", false);

  //Initialize Constraints
  // addRLConstraints(); // Add constraints specific to the RL policy // DONT WORK YET
  selfCollisionConstraint->setCollisionsDampers(solver(), {1.2, 200.0});
  solver().removeConstraintSet(dynamicsConstraint);
  dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(
    new mc_solver::DynamicsConstraint(robots(), 0, {diPercent, dsPercent, 0.0, 1.2, 200.0}, velPercent, true));
  solver().addConstraintSet(dynamicsConstraint);

  std::unique_ptr<mc_solver::ContactConstraint> contactConstraintTest;
  contactConstraintTest =std::make_unique<mc_solver::ContactConstraint>(timeStep, mc_solver::ContactConstraint::ContactType::Acceleration);
  solver().addConstraintSet(contactConstraintTest);
  
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

// void RLController::manageContacts()
// {
//   addContact(const Contact &c)
// }

bool RLController::run()
{
  counter += timeStep;
  leftAnklePos = robot().mbc().bodyPosW[robot().bodyIndexByName("left_ankle_link")].translation();
  rightAnklePos = robot().mbc().bodyPosW[robot().bodyIndexByName("right_ankle_link")].translation();
  ankleDistanceNorm = (leftAnklePos - rightAnklePos).norm();
  computeLimits();

  if(forceToVelControl) forceToVelocityControl();

  auto & real_robot = realRobot(robots()[0].name());

  auto qIn = real_robot.mbc().q;
  auto alphaIn = real_robot.mbc().alpha;
  // auto tauIn = real_robot.mbc().jointTorque;
  floatingBase_qIn = rbd::paramToVector(robot().mb(), qIn);
  floatingBase_alphaIn = rbd::dofToVector(robot().mb(), alphaIn);
  Eigen::MatrixXd Kp_inv = current_kp.cwiseInverse().asDiagonal();
  auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
  auto tau_ext = extTorqueSensor.torques();

  bool run = mc_control::fsm::Controller::run(mc_solver::FeedbackType::ClosedLoopIntegrateReal);
  robot().forwardKinematics();
  robot().forwardVelocity();
  robot().forwardAcceleration();
  if(!useQP) // Run RL without taking account of the QP
  {
    q_cmd = q_rl; // Use the RL position as the commanded position
    tau_cmd = kp_vector.cwiseProduct(q_rl - currentPos) - kd_vector.cwiseProduct(currentVel);
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
  // utils_.stopInferenceThread();
}

void RLController::tasksComputation(Eigen::VectorXd & currentTargetPosition)
{
  auto & robot = robots()[0];
  auto & real_robot = realRobot(robots()[0].name());

  auto q = real_robot.encoderValues();
  currentPos = Eigen::VectorXd::Map(q.data(), q.size());
  auto vel = real_robot.encoderVelocities();
  currentVel = Eigen::VectorXd::Map(vel.data(), vel.size());
  auto tau = real_robot.jointTorques();
  currentTau = Eigen::VectorXd::Map(tau.data(), tau.size());

  if(controlledByRL) tau_d = kp_vector.cwiseProduct(currentTargetPosition - currentPos) - kd_vector.cwiseProduct(currentVel);
  else tau_d = high_kp_vector.cwiseProduct(currentTargetPosition - currentPos) - high_kd_vector.cwiseProduct(currentVel);
  
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
      
      auto extTorqueSensor = robot.device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
      Eigen::VectorXd tau_d_w_floating_base = Eigen::VectorXd::Zero(robot.mb().nrDof());
      tau_d_w_floating_base.tail(dofNumber) = tau_d.tail(dofNumber);
      Eigen::VectorXd content = tau_d_w_floating_base - Cg_w_floatingBase; // Add the external torques to the desired torques
      if(!compensateExternalForces) content += extTorqueSensor.torques();
      
      Eigen::VectorXd refAccel_w_floating_base = M_w_floatingBase.llt().solve(content);
      refAccel = refAccel_w_floating_base.tail(dofNumber); // Exclude the floating base part
      break;
    }
    default:
      mc_rtc::log::error("Invalid task type: {}", taskType);
      return;
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
  if(controlledByRL) robot().mbc().alpha = alpha; // For RL policy qdot ref = 0
  // Update joint torques for torque control
  robot().mbc().jointTorque = tau;
  // Both are always updated despite they are not used by the robot
  // They are still used by the QP
}

void RLController::computeInversePD()
{
  // Using QP (TorqueTask or ForwardDynamics Task):  
  ddot_qp_w_floatingBase = rbd::dofToVector(robot().mb(), robot().mbc().alphaD);
  ddot_qp = ddot_qp_w_floatingBase.tail(dofNumber); // Exclude the floating base part
  auto & real_robot = realRobot(robots()[0].name());

  rbd::ForwardDynamics fd(real_robot.mb());
  fd.computeH(real_robot.mb(), real_robot.mbc());
  fd.computeC(real_robot.mb(), real_robot.mbc());
  Eigen::MatrixXd M_w_floatingBase = fd.H();
  Eigen::VectorXd Cg_w_floatingBase = fd.C();

  auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
  Eigen::VectorXd tau_cmd_w_floatingBase = M_w_floatingBase*ddot_qp_w_floatingBase + Cg_w_floatingBase - extTorqueSensor.torques();
  tau_cmd = tau_cmd_w_floatingBase.tail(dofNumber);

  qddot_rl_simulatedMeasure = M_w_floatingBase.llt().solve(tau_rl + extTorqueSensor.torques() - Cg_w_floatingBase).tail(dofNumber);
  qdot_rl_simulatedMeasure = currentVel + qddot_rl_simulatedMeasure*timeStep;
  q_rl_simulatedMeasure += qdot_rl_simulatedMeasure*timeStep;

  Eigen::MatrixXd Kp_inv = current_kp.cwiseInverse().asDiagonal();

  q_cmd = currentPos + Kp_inv*(tau_cmd + current_kd.cwiseProduct(currentVel)); // Inverse PD control to get the commanded position <=> RL position control
}

void RLController::computeRLStateSimulated()
{
  auto & real_robot = realRobot(robots()[0].name());
  rbd::ForwardDynamics fd(real_robot.mb());
  fd.computeH(real_robot.mb(), real_robot.mbc());
  fd.computeC(real_robot.mb(), real_robot.mbc());
  Eigen::MatrixXd M_w_floatingBase = fd.H();
  Eigen::VectorXd Cg_w_floatingBase = fd.C();

  auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
  tau_rl = kp_vector.cwiseProduct(q_rl - currentPos) - kd_vector.cwiseProduct(currentVel);
  Eigen::VectorXd tau_rl_w_floating_base = Eigen::VectorXd::Zero(robot().mb().nrDof());
  tau_rl_w_floating_base.tail(dofNumber) = tau_rl;
  Eigen::VectorXd content = tau_rl_w_floating_base + extTorqueSensor.torques() - Cg_w_floatingBase; // Add the external torques to the desired torques

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
  logger().addLogEntry("RLController_kp", [this]() { return current_kp; });
  logger().addLogEntry("RLController_kd", [this]() { return current_kd; });
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
  logger().addLogEntry("RLController_controlledByRL", [this]() { return controlledByRL; });
  logger().addLogEntry("RLController_taskType", [this]() { return taskType; });

  // RL Controller
  logger().addLogEntry("RLController_q_lim_upper", [this]() { return jointLimitsPos_upper; });
  logger().addLogEntry("RLController_q_lim_lower", [this]() { return jointLimitsPos_lower; });
  logger().addLogEntry("RLController_qdot_lim_upper", [this]() { return jointLimitsVel_upper; });
  logger().addLogEntry("RLController_qdot_lim_lower", [this]() { return jointLimitsVel_lower; });
  logger().addLogEntry("RLController_qdot_limHard_upper", [this]() { return jointLimitsHardVel_upper; });
  logger().addLogEntry("RLController_qdot_limHard_lower", [this]() { return jointLimitsHardVel_lower; });
  logger().addLogEntry("RLController_ankleDistanceNorm", [this]() { return ankleDistanceNorm; });
  logger().addLogEntry("RLController_limitBreached_q_soft_upper", [this]() { return limitBreached_q_soft_upper; });
  logger().addLogEntry("RLController_limitBreached_q_soft_lower", [this]() { return limitBreached_q_soft_lower; });
  logger().addLogEntry("RLController_limitBreached_q_hard_upper", [this]() { return limitBreached_q_hard_upper; });
  logger().addLogEntry("RLController_limitBreached_q_hard_lower", [this]() { return limitBreached_q_hard_lower; });
  logger().addLogEntry("RLController_limitBreached_qdot_soft_upper", [this]() { return limitBreached_qDot_soft_upper; });
  logger().addLogEntry("RLController_limitBreached_qdot_soft_lower", [this]() { return limitBreached_qDot_soft_lower; });
  logger().addLogEntry("RLController_limitBreached_qdot_hard_upper", [this]() { return limitBreached_qDot_hard_upper; });
  logger().addLogEntry("RLController_limitBreached_qdot_hard_lower", [this]() { return limitBreached_qDot_hard_lower; });
  logger().addLogEntry("RLController_limitBreached_tau_hard_upper", [this]() { return limitBreached_tau_upper; });
  logger().addLogEntry("RLController_limitBreached_tau_hard_lower", [this]() { return limitBreached_tau_lower; });
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

  logger().addLogEntry("RLController_forceVel", [this]() { return forceVel; });
  logger().addLogEntry("RLController_forceVel_filtered", [this]() { return forceVel_filtered; });
}

void RLController::addGui()
{
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::Checkbox("Compensate External Forces", compensateExternalForces));
  // Add a button to change the velocity command
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::ArrayInput("Velocity Command RL", {"X", "Y", "Yaw"}, velCmdRL_));
  
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::Checkbox("Force to Velocity Control", forceToVelControl));
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::NumberInput("Velocity Max (m/s)", velMax));
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::NumberInput("Yaw Rate Max (rad/s)", yawRateMax));
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::NumberInput("Fx Scale", fxScale));
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::NumberInput("Fy Scale", fyScale));
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::NumberInput("Tauy Scale", tauyScale));
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::NumberInput("Fx Deadzone", fxDeadZone));
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::NumberInput("Fy Deadzone", fyDeadZone));
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::NumberInput("Tauy Deadzone", tauyDeadZone));
  gui()->addElement({"FSM", "Options"},
  mc_rtc::gui::NumberInput("Force Vel Filter Alpha", forceVel_filter_alpha));

  // gui()->addPlot("Velocity Command RL", 
  //   mc_rtc::gui::plot::X("t", [this]() { return counter; }),
  //   mc_rtc::gui::plot::Y("vx", [this]() { return velCmdRL_[0]; }, mc_rtc::gui::Color::Red),
  //   mc_rtc::gui::plot::Y("vy", [this]() { return velCmdRL_[1]; }, mc_rtc::gui::Color::Green),
  //   mc_rtc::gui::plot::Y("vyaw", [this]() { return velCmdRL_[2]; }, mc_rtc::gui::Color::Blue)
  // );
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

  auto & real_robot = realRobot(robots()[0].name());

  leftAnklePos = real_robot.collisionTransform("left_ankle_link").translation();
  rightAnklePos = real_robot.collisionTransform("right_ankle_link").translation();
  ankleDistanceNorm = (leftAnklePos - rightAnklePos).norm();

  jointLimitsHardPos_upper = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardPos_lower = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardVel_upper = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardVel_lower = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardTau_upper = Eigen::VectorXd::Zero(dofNumber);
  jointLimitsHardTau_lower = Eigen::VectorXd::Zero(dofNumber);

  limitBreached_q_soft_upper = Eigen::VectorXd::Zero(dofNumber);
  limitBreached_q_soft_lower = Eigen::VectorXd::Zero(dofNumber);
  limitBreached_q_hard_upper = Eigen::VectorXd::Zero(dofNumber);
  limitBreached_q_hard_lower = Eigen::VectorXd::Zero(dofNumber);
  limitBreached_qDot_soft_upper = Eigen::VectorXd::Zero(dofNumber);
  limitBreached_qDot_soft_lower = Eigen::VectorXd::Zero(dofNumber);
  limitBreached_qDot_hard_upper = Eigen::VectorXd::Zero(dofNumber);
  limitBreached_qDot_hard_lower = Eigen::VectorXd::Zero(dofNumber);
  limitBreached_tau_upper = Eigen::VectorXd::Zero(dofNumber);
  limitBreached_tau_lower = Eigen::VectorXd::Zero(dofNumber);

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
  high_kp_vector = Eigen::VectorXd::Zero(dofNumber);
  high_kd_vector = Eigen::VectorXd::Zero(dofNumber);
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
            q_rl[i] = t[0];
            q_zero_vector[i] = t[0];
            torque_target[joint_name] = {0.0};
            mc_rtc::log::info("[RLController] Joint {}: currentTargetPosition {}, kp {}, kd {}", joint_name, q_rl[i], kp_vector[i], kd_vector[i]);
            i++;
        }
      }
  }
  current_kp = high_kp_vector;
  current_kd = high_kd_vector;
  solver().removeTask(FSMPostureTask);
  datastore().make<std::string>("ControlMode", "Torque");
  if(!datastore().has("anchorFrameFunction"))
  {
    datastore().make_call("anchorFrameFunction", [this](const mc_rbdyn::Robot & robot) {return createContactAnchor(robot);});
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

  // Velocity from the integration of the acceleration provided by the external forces
  forceVel = Eigen::VectorXd::Zero(robot().mb().nrDof());
  forceVel_filtered = Eigen::VectorXd::Zero(robot().mb().nrDof());
}

void RLController::initializeRLPolicy(const mc_rtc::Configuration & config)
{
  auto & real_robot = realRobot(robots()[0].name());

  baseAngVel = real_robot.bodyVelW("pelvis").angular();
  Eigen::Matrix3d baseRot = real_robot.bodyPosW("pelvis").rotation();
  rpy = mc_rbdyn::rpyFromMat(baseRot);
    
  mc_rtc::log::info("[RLController] Posture target initialized with {} joints", dofNumber); 

  // Initialize reference position and last actions for action blending
  a_before_vector = Eigen::VectorXd::Zero(dofNumber);
  a_vector = Eigen::VectorXd::Zero(dofNumber);
  legPos = Eigen::VectorXd::Zero(10);
  legVel = Eigen::VectorXd::Zero(10);
  legAction = Eigen::VectorXd::Zero(10);

  baseAngVel_prev = Eigen::Vector3d::Zero();
  rpy_prev = Eigen::Vector3d::Zero();
  legPos_prev = Eigen::VectorXd::Zero(10);
  legVel_prev = Eigen::VectorXd::Zero(10);
  legAction_prev = Eigen::VectorXd::Zero(10);

  baseAngVel_prev_prev = Eigen::Vector3d::Zero();
  rpy_prev_prev = Eigen::Vector3d::Zero();
  legPos_prev_prev = Eigen::VectorXd::Zero(10);
  legVel_prev_prev = Eigen::VectorXd::Zero(10);
  legAction_prev_prev = Eigen::VectorXd::Zero(10);

  a_simuOrder = Eigen::VectorXd::Zero(dofNumber);

  mc_rtc::log::info("Reference position initialized with {} joints", q_zero_vector.size());
  q_rl = q_zero_vector;  // Start with reference position
  
  useAsyncInference_ = config("use_async_inference", true);
  mc_rtc::log::info("Async RL inference: {}", useAsyncInference_ ? "enabled" : "disabled");
  currentAction_ = Eigen::VectorXd::Zero(dofNumber);
  latestAction_ = Eigen::VectorXd::Zero(dofNumber);
  
  // Initialize new observation components
  velCmdRL_ = Eigen::Vector3d::Zero();  // Default command (x, y, yaw)
  // Values for the aggressive trot
  // velCmdRL_(0) = 1.09;
  // velCmdRL_(1) = 0.6;
  // velCmdRL_(2) = 0.5;
  phase_ = 0.0;  // Phase for periodic gait
  startPhase_ = std::chrono::steady_clock::now();  // For phase calculation
  
  // Initialize external force application (hardcoded for testing)
  externalForceEnabled_ = false;
  pelvisForceEnabled_ = false;
  pelvisForce_ = Eigen::Vector3d(0.0, 0.10, -100.0);  // Default 100N downward force for testing
  mc_rtc::log::info("[RLController] External force application initialized with test force: [{:.1f}, {:.1f}, {:.1f}] N", 
                   pelvisForce_.x(), pelvisForce_.y(), pelvisForce_.z());
  mc_rtc::log::info("[RLController] Robot name: '{}', Available bodies:", robot().name());
  for(const auto & body : robot().mb().bodies()) {
    mc_rtc::log::info("[RLController]   Body: '{}'", body.name());
  }
  
  std::string policyPath = config("policy_path", std::string(""));
  if(policyPath.empty())
    policyPath = "policy.onnx"; // Default policy path if not specified in config

  mc_rtc::log::info("Loading RL policy from: {}", policyPath);
  try {
    rlPolicy_ = std::make_unique<RLPolicyInterface>(policyPath);
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
  // Check if the current gains are close to the low gains
  bool lowGain = ((current_kp - kp_vector).norm() < tol) && ((current_kd - kd_vector).norm() < tol);
  bool highGain = !lowGain;
  mc_rtc::log::info("[RLController] current_kp: {}", current_kp.transpose());
  mc_rtc::log::info("[RLController] current_kd: {}", current_kd.transpose());
  mc_rtc::log::info("[RLController] kp_vector: {}", kp_vector.transpose());
  mc_rtc::log::info("[RLController] kd_vector: {}", kd_vector.transpose());
  mc_rtc::log::info("[RLController] isHighGain: {}", highGain);
  return highGain;
}

void RLController::initializeState(bool torque_control, int task_type, bool controlled_by_rl)
{
  if(torque_control) datastore().get<std::string>("ControlMode") = "Torque";
  else datastore().get<std::string>("ControlMode") = "Position";

  if (!datastore().call<bool>("EF_Estimator::isActive")) {
    datastore().call("EF_Estimator::toggleActive");
  }

  useQP = true;
  if(task_type == PURE_RL) useQP = false;
  else taskType = task_type;
  controlledByRL = controlled_by_rl;
  if(controlledByRL)
  {
    // Set low gains for RL
    if(isHighGain()) setPDGains(kp_vector, kd_vector);
    tasksComputation(q_rl);
  }
  else
  {
    // Set high gains for model-based control
    if(!isHighGain()) setPDGains(high_kp_vector, high_kd_vector);
    tasksComputation(q_zero_vector);
  }
}

void RLController::computeLimits()
{
  bool hardBreached = false;
  double epsilon = 1e-5;
  for (size_t i = 0; i < dofNumber; ++i)
  {
    // q lim
    hardBreached = false;
    limitBreached_q_hard_upper(i) = 0.0;
    if(currentPos(i) > jointLimitsHardPos_upper(i) + epsilon)
    {
      limitBreached_q_hard_upper(i) = 1.0;
      mc_rtc::log::info("t= {}s; Joint {} position upper hard limit breached: currentPos = {}, limit = {}", counter, jointNames[i], currentPos(i), jointLimitsHardPos_upper(i));
      hardBreached = true;
    }

    limitBreached_q_soft_upper(i) = 0.0;
    if(currentPos(i) > jointLimitsPos_upper(i) + epsilon)
    {
      limitBreached_q_soft_upper(i) = 1.0;
      if(!hardBreached) mc_rtc::log::info("t= {}s; Joint {} position upper soft limit breached: currentPos = {}, limit = {}", counter, jointNames[i], currentPos(i), jointLimitsPos_upper(i));
    }
      
    hardBreached = false;
    limitBreached_q_hard_lower(i) = 0.0;
    if(currentPos(i) < jointLimitsHardPos_lower(i) - epsilon)
    {
      limitBreached_q_hard_lower(i) = 1.0;
      mc_rtc::log::info("t= {}s; Joint {} position lower hard limit breached: currentPos = {}, limit = {}", counter, jointNames[i], currentPos(i), jointLimitsHardPos_lower(i));
      hardBreached = true;
    }

    limitBreached_q_soft_lower(i) = 0.0;
    if(currentPos(i) < jointLimitsPos_lower(i) - epsilon)
    {
      limitBreached_q_soft_lower(i) = 1.0;
      if(!hardBreached) mc_rtc::log::info("t= {}s; Joint {} position lower soft limit breached: currentPos = {}, limit = {}", counter, jointNames[i], currentPos(i), jointLimitsPos_lower(i));
    }

    //qdot lim
    hardBreached = false;
    limitBreached_qDot_hard_upper(i) = 0.0;
    if(currentVel(i) > jointLimitsHardVel_upper(i) + epsilon)
    {
      limitBreached_qDot_hard_upper(i) = 1.0;
      mc_rtc::log::info("t= {}s; Joint {} velocity upper hard limit breached: currentVel = {}, limit = {}", counter, jointNames[i], currentVel(i), jointLimitsHardVel_upper(i));
      hardBreached = true;
    }
      
    limitBreached_qDot_soft_upper(i) = 0.0;
    if(currentVel(i) > jointLimitsVel_upper(i) + epsilon)
    {
      limitBreached_qDot_soft_upper(i) = 1.0;
      if(!hardBreached) mc_rtc::log::info("t= {}s; Joint {} velocity upper soft limit breached: currentVel = {}, limit = {}", counter, jointNames[i], currentVel(i), jointLimitsVel_upper(i));
    }
      
    hardBreached = false;
    limitBreached_qDot_hard_lower(i) = 0.0;
    if(currentVel(i) < jointLimitsHardVel_lower(i) - epsilon)
    {
      limitBreached_qDot_hard_lower(i) = 1.0;
      mc_rtc::log::info("t= {}s; Joint {} velocity lower hard limit breached: currentVel = {}, limit = {}", counter, jointNames[i], currentVel(i), jointLimitsHardVel_lower(i));
      hardBreached = true;
    }

    limitBreached_qDot_soft_lower(i) = 0.0;
    if(currentVel(i) < jointLimitsVel_lower(i) - epsilon)
    {
      limitBreached_qDot_soft_lower(i) = 1.0;
      if(!hardBreached) mc_rtc::log::info("t= {}s; Joint {} velocity lower soft limit breached: currentVel = {}, limit = {}", counter, jointNames[i], currentVel(i), jointLimitsVel_lower(i));
    }
      
    // tau lim
    limitBreached_tau_upper(i) = 0.0;
    if(tau_cmd(i) > jointLimitsHardTau_upper(i) + epsilon)
    {
      limitBreached_tau_upper(i) = 1.0;
      mc_rtc::log::info("t= {}s; Joint {} torque upper hard limit breached: currentTau = {}, limit = {}", counter, jointNames[i], tau_cmd(i), jointLimitsHardTau_upper(i));
    }

    limitBreached_tau_lower(i) = 0.0;
    if(tau_cmd(i) < jointLimitsHardTau_lower(i) - epsilon)
    {
      limitBreached_tau_lower(i) = 1.0;
      mc_rtc::log::info("t= {}s; Joint {} torque lower hard limit breached: currentTau = {}, limit = {}", counter, jointNames[i], tau_cmd(i), jointLimitsHardTau_lower(i));
    }
  }
}

std::pair<sva::PTransformd, Eigen::Vector3d> RLController::createContactAnchor(const mc_rbdyn::Robot & anchorRobot)
{
  sva::PTransformd X_foot_r = anchorRobot.bodyPosW("right_ankle_link");
  sva::PTransformd X_foot_l = anchorRobot.bodyPosW("left_ankle_link");

  sva::MotionVecd v_foot_r = anchorRobot.bodyVelW("right_ankle_link");
  sva::MotionVecd v_foot_l = anchorRobot.bodyVelW("left_ankle_link");

  auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
  // double tau_ext_knee_r = extTorqueSensor.torques()[robot().jointIndexByName("right_knee_joint")];
  // double tau_ext_knee_l = extTorqueSensor.torques()[robot().jointIndexByName("left_knee_joint")];
  double tau_ext_knee_r =  abs(extTorqueSensor.torques()[8+6]);
  double tau_ext_knee_l =  abs(extTorqueSensor.torques()[3+6]);
  if(tau_ext_knee_r + tau_ext_knee_l < 1e-6)
  {
    tau_ext_knee_l = 1;
    tau_ext_knee_r = 1;
  }
  double leftFootRatio = tau_ext_knee_l/(tau_ext_knee_r+tau_ext_knee_l);
                              
  Eigen::VectorXd w_r = X_foot_r.translation();//* tau_ext_knee_r/(tau_ext_knee_r+tau_ext_knee_l);
  Eigen::VectorXd w_l = X_foot_l.translation(); //* tau_ext_knee_l/(tau_ext_knee_r+tau_ext_knee_l);
  Eigen::VectorXd contact_anchor = (w_r * (1 - leftFootRatio) + w_l * leftFootRatio)  ;
  Eigen::VectorXd anchor_vel = (v_foot_r.linear() * (1 - leftFootRatio) + v_foot_l.linear() * leftFootRatio);
  sva::PTransformd contact_anchor_tf(Eigen::Matrix3d::Identity(), contact_anchor); 

  return {contact_anchor_tf, anchor_vel};
}

void RLController::addRLConstraints()
{
  Eigen::VectorXd torqueLimManiskillOrder(19);
  torqueLimManiskillOrder << 150.0, 150.0, 150.0, 150.0, 150.0, 
                              30.0, 30.0, 150.0, 150.0, 30.0, 
                              30.0, 150.0, 150.0, 30.0, 30.0, 
                              30.0, 30.0, 30.0, 30.0;

  Eigen::VectorXd qLimManiskillOrder_lower(19);
  qLimManiskillOrder_lower << -0.48, -0.48, -0.35, -0.35, -0.35, 
                              -1.05, -1.05, -2.58, -2.58, -0.39, 
                              -0.39, 0.05,  0.05,  -1.35, -1.35, 
                              -0.92, -0.92, -1.30, -1.30;

  Eigen::VectorXd qLimManiskillOrder_upper(19);
  qLimManiskillOrder_upper << 0.48,  0.48,  0.35,  0.35,  0.35,  
                              1.05,  1.05,  2.58,  2.58,  0.39,  
                              0.39, 2.10, 2.10,  1.35,  1.35,  
                              0.57, 0.57,  1.30,  1.30;

  Eigen::VectorXd maniskillToMcRtcIdx_(19);
  maniskillToMcRtcIdx_ << 0, 3, 7, 11, 15, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 6, 10, 14, 18;

  // Convert to simulator order
  Eigen::VectorXd torqueLim_simuOrder = Eigen::VectorXd::Zero(19);
  Eigen::VectorXd qLim_simuOrder_lower = Eigen::VectorXd::Zero(19);
  Eigen::VectorXd qLim_simuOrder_upper = Eigen::VectorXd::Zero(19);
        
  for(size_t i = 0; i < maniskillToMcRtcIdx_.size(); ++i)
  {
    int simuIdx = maniskillToMcRtcIdx_[i];
    if(simuIdx != -1)
    {
      torqueLim_simuOrder(i) = torqueLimManiskillOrder(simuIdx);
      qLim_simuOrder_lower(i) = qLimManiskillOrder_lower(simuIdx);
      qLim_simuOrder_upper(i) = qLimManiskillOrder_upper(simuIdx);
    }
  }

  for (size_t i = 0 ; i < robot().refJointOrder().size() ; i++)
  {
    const std::string & jname = robot().refJointOrder()[i];
    auto mcJointId = robot().jointIndexByName(jname);
    if (robot().mbc().q[mcJointId].empty())
      continue;
    
    robot().ql().at(mcJointId)[0] = qLim_simuOrder_lower(i);
    robot().qu().at(mcJointId)[0] = qLim_simuOrder_upper(i);
    robot().tl().at(mcJointId)[0] = -torqueLim_simuOrder(i);
    robot().tu().at(mcJointId)[0] = torqueLim_simuOrder(i);
  }
}

void RLController::forceToVelocityControl()
{
  auto extTorqueSensor = robot().device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");
  auto & real_robot = realRobot(robots()[0].name());
  rbd::ForwardDynamics fd(real_robot.mb());
  fd.computeH(real_robot.mb(), real_robot.mbc());
  Eigen::MatrixXd M_w_floatingBase = fd.H();
    
  Eigen::VectorXd refAccel_w_floating_base = M_w_floatingBase.llt().solve(extTorqueSensor.torques());
  forceVel += refAccel_w_floating_base * timeStep;
  forceVel_filtered = forceVel_filter_alpha * forceVel_filtered + (1 - forceVel_filter_alpha) * forceVel;
  // forceVel = forceVel_filtered;

  rbd::Jacobian jac = rbd::Jacobian(robot().mb(), "left_elbow_link");
  // Eigen::MatrixXd J_full = jac.jacobian(real_robot.mb(), real_robot.mbc());
  // // Partition J_full = [J_base (6x6) , J_act (6 x n_a)]
  // const int cols = J_full.cols();
  // const int n_base = 6;
  // const int n_act = cols - n_base;
  // Eigen::MatrixXd J_base = J_full.block(0, 0, 6, n_base);   // if you need it
  // Eigen::MatrixXd J_act  = J_full.block(0, n_base, 6, n_act); // 6 x n_a

  // Eigen::VectorXd tau_full = extTorqueSensor.torques(); // (6 + n_a) x 1
  // // Least squares estimate of wrench W (6x1)
  // Eigen::MatrixXd JJt = J_full * J_full.transpose(); // 6x6
  // double lambda = 1e-6;
  // JJt += lambda * Eigen::MatrixXd::Identity(6,6);
  // Eigen::VectorXd rhs = J_full * tau_full;
  // Eigen::VectorXd W = JJt.ldlt().solve(rhs);

  auto jTranspose = jac.jacobian(real_robot.mb(), real_robot.mbc());
  jTranspose.transposeInPlace();
  Eigen::VectorXd leftArmForce = jTranspose.completeOrthogonalDecomposition().solve(extTorqueSensor.torques());

  // double fx = W[3];
  // double fy = W[4];
  // double tauy = W[1];

  // double fx = forceVel_filtered[3];
  // double fy = forceVel_filtered[4];
  // double tauy = forceVel_filtered[1];

  double fx = leftArmForce[3];
  double fy = leftArmForce[4];
  double tauy = leftArmForce[1];

  // double fx = extTorqueSensor.torques()[0];
  // double fy = extTorqueSensor.torques()[1];
  // double tauy = extTorqueSensor.torques()[4];

  double velx = 0.0;
  if(fx> fxDeadZone) velx = (fx - fxDeadZone) * fxScale;
  if(velx > velMax) velx = velMax;
  if(velx < -velMax) velx = -velMax;

  double vely = 0.0;
  if(fy> fyDeadZone) vely = (fy - fyDeadZone) * fyScale;
  if(vely > velMax) vely = velMax;
  if(vely < -velMax) vely = -velMax;

  double yawRate = 0.0;
  if(tauy> tauyDeadZone) yawRate = (tauy - tauyDeadZone) * tauyScale;
  if(yawRate > yawRateMax) yawRate = yawRateMax;
  if(yawRate < -yawRateMax) yawRate = -yawRateMax;

  velCmdRL_(0) = velx;
  velCmdRL_(1) = vely;
  velCmdRL_(2) = yawRate;
}