#include "RLController.h"
#include <Eigen/src/Core/VectorBlock.h>
#include <RBDyn/MultiBodyConfig.h>
#include <SpaceVecAlg/SpaceVecAlg>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <mc_rtc/gui/ArrayInput.h>
#include <mc_rtc/logging.h>
#include <mc_rbdyn/configuration_io.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>


RLController::RLController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(rm, dt, config, Backend::TVM)
{
  logTiming_ = config("log_timing");
  timingLogInterval_ = config("timing_log_interval");
  useQP = config("use_QP", true);
  isTorqueControl = config("is_torque_control", false);

  //Initialize Constraints
  selfCollisionConstraint->setCollisionsDampers(solver(), {1.2, 200.0});
  solver().removeConstraintSet(dynamicsConstraint);
  dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(
    new mc_solver::DynamicsConstraint(robots(), 0, {diPercent, dsPercent, 0.0, 1.2, 200.0}, velPercent, true));
  solver().addConstraintSet(dynamicsConstraint);

  // Initialize Task
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
  counter += timeStep;
  leftAnklePos = robot().mbc().bodyPosW[robot().bodyIndexByName("left_ankle_link")].translation();
  rightAnklePos = robot().mbc().bodyPosW[robot().bodyIndexByName("right_ankle_link")].translation();
  ankleDistanceNorm = (leftAnklePos - rightAnklePos).norm();

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

  tau_d = kp_vector.cwiseProduct(currentTargetPosition - currentPos) - kd_vector.cwiseProduct(currentVel);
   
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
}

void RLController::addGui()
{
  gui()->addElement({"RLController", "Options"},
  mc_rtc::gui::Checkbox("Compensate External Forces", compensateExternalForces));
  // Add a button to change the velocity command
  gui()->addElement({"RLController", "Policy"},
  mc_rtc::gui::ArrayInput("Velocity Command RL", {"X", "Y", "Yaw"}, velCmdRL_));

  // Add a dropdown to select policy
  gui()->addElement(
    {"RLController", "Policy"},
    mc_rtc::gui::ComboInput(
      "Policy Path",
      policyPaths,
      [this]() -> const std::string & { 
        return policyPaths[currentPolicyIndex]; 
      },
      [this](const std::string & selected) { 
        // Find the index of the selected policy
        auto it = std::find(policyPaths.begin(), policyPaths.end(), selected);
        if(it != policyPaths.end()) {
          currentPolicyIndex = std::distance(policyPaths.begin(), it);
          mc_rtc::log::info("Selected policy [{}]: {}", currentPolicyIndex, selected);
          // Loading the new policy
          rlPolicy_ = std::make_unique<RLPolicyInterface>(policyPaths[currentPolicyIndex]);
        }
      }
    )
  );
}

void RLController::initializeRobot(const mc_rtc::Configuration & config)
{
  // H1 joints in mc_rtc/URDF order (based on unitree_sdk2 reorder_obs function)
  auto mcRtcJointsOrder = config("mc_rtc_joints_order");

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
  
  // Load policy paths from config
  policyPaths = config("policy_path", std::vector<std::string>{"example_walking_h1.onnx"});
  currentPolicyIndex = 0;  // Start with first policy
  
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

  std::string simulator = config("Simulator", std::string(""));
  if (simulator.empty())
  {
    mc_rtc::log::warning("Simulator not set, using default handling");
    policySimulatorHandling_ = std::make_unique<PolicySimulatorHandling>();
  }
  else
  {
    mc_rtc::log::info("Using {} handling", simulator);
    policySimulatorHandling_ = std::make_unique<PolicySimulatorHandling>(simulator);
  }

  // get list of used joints from config
  usedJoints_mcRtcOrder = config("Used_joints_index", std::vector<int>{});
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

void RLController::initializeState()
{
  if(isTorqueControl) datastore().get<std::string>("ControlMode") = "Torque";
  else datastore().get<std::string>("ControlMode") = "Position";

  if (!datastore().call<bool>("EF_Estimator::isActive")) {
    datastore().call("EF_Estimator::toggleActive");
  }

  // Set low gains for RL
  if(isHighGain()) setPDGains(kp_vector, kd_vector);
  tasksComputation(q_rl);
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