#include "RLController.h"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <mc_rtc/logging.h>
#include <mc_rbdyn/configuration_io.h>
#include <chrono>

RLController::RLController(mc_rbdyn::RobotModulePtr rm, double dt, 
                           const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(rm, dt, config, Backend::TVM)
{
  // Add constraints
  // contactConstraintTest =std::make_unique<mc_solver::ContactConstraint>(timeStep, mc_solver::ContactConstraint::ContactType::Acceleration);
  // solver().addConstraintSet(contactConstraintTest);

  // selfCollisionConstraint->setCollisionsDampers(solver(), {1.8, 70.0});
  
  // dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(
  //     new mc_solver::DynamicsConstraint(
  //         robots(), 0, {0.1, 0.01, 0.0, 1.8, 70.0}, 0.9, true));
  // solver().addConstraintSet(dynamicsConstraint);

  solver().removeTask(getPostureTask(robot().name()));

  datastore().make<std::string>("ControlMode", "Torque");
  mc_rtc::log::success("RLController init");
}

RLController::~RLController()
{
  stopInferenceThread();
  mc_rtc::log::info("RLController destroyed");
}

bool RLController::run()
{
  // return mc_control::fsm::Controller::run();
  return mc_control::fsm::Controller::run(mc_solver::FeedbackType::ClosedLoopIntegrateReal);
}

void RLController::reset(const mc_control::ControllerResetData & reset_data)
{
  mc_control::fsm::Controller::reset(reset_data);
  
  initializeAllJoints();
  initializeImpedanceGains();
  pastAction_ = Eigen::VectorXd::Zero(19);
  
  useAsyncInference_ = config()("use_async_inference", true);
  mc_rtc::log::info("Async RL inference: {}", useAsyncInference_ ? "enabled" : "disabled");
  
  shouldStopInference_ = false;
  newObservationAvailable_ = false;
  newActionAvailable_ = false;
  currentObservation_ = Eigen::VectorXd::Zero(35);
  currentAction_ = Eigen::VectorXd::Zero(19);
  latestAction_ = Eigen::VectorXd::Zero(19);
  
  std::string policyPath = config()("policy_path", std::string(""));
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
  postureTask_ =  std::make_shared<mc_tasks::PostureTask>(
      solver(), robot().robotIndex(), 1, 1);
  postureTask_->stiffness(100.0);
  solver().addTask(postureTask_);
  
  if(useAsyncInference_)
  {
    startInferenceThread();
  }
  
  mc_rtc::log::success("RLController reset completed");
}

void RLController::initializeAllJoints()
{
  // H1 joints in mc_rtc/URDF order (based on unitree_sdk2 reorder_obs function)
  allJoints_ = {
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
  
  // leg joints (first 10)
  legJoints_ = std::vector<std::string>(allJoints_.begin(), allJoints_.begin() + 10);
  
  maniskillToMcRtcIdx_ = {0, 3, 7, 11, 15, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 6, 10, 14, 18};
  mcRtcToManiskillIdx_ = {0, 5, 10, 1, 6, 11, 15, 2, 7, 12, 16, 3, 8, 13, 17, 4, 9, 14, 18};
}

void RLController::initializeImpedanceGains()
{
  kp_ = Eigen::VectorXd(19);
  kd_ = Eigen::VectorXd(19);
  
  kp_.segment(0, 5) << 100.0, 100.0, 100.0, 100.0, 20.0;  // Left leg
  kd_.segment(0, 5) << 10.0, 10.0, 10.0, 10.0, 4.0;
  
  kp_.segment(5, 5) << 100.0, 100.0, 100.0, 100.0, 20.0;  // Right leg  
  kd_.segment(5, 5) << 10.0, 10.0, 10.0, 10.0, 4.0;
  
  // Torso
  kp_(10) = 100.0; kd_(10) = 10.0;
  
  kp_.segment(11, 4) << 100.0, 100.0, 100.0, 100.0;  // Left arm
  kd_.segment(11, 4) << 5.0, 5.0, 5.0, 5.0;
  
  kp_.segment(15, 4) << 100.0, 100.0, 100.0, 100.0;  // Right arm
  kd_.segment(15, 4) << 5.0, 5.0, 5.0, 5.0;
}

Eigen::VectorXd RLController::reorderObservationToManiskill(const Eigen::VectorXd & obs)
{  
  if(obs.size() != 19) {
    mc_rtc::log::error("Observation reordering expects 19 joints, got {}", obs.size());
    return obs;
  }
  
  Eigen::VectorXd reordered(19);
  for(int i = 0; i < 19; i++) {
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

Eigen::VectorXd RLController::reorderActionFromManiskill(const Eigen::VectorXd & action)
{  
  if(action.size() != 19) {
    mc_rtc::log::error("Action reordering expects 19 joints, got {}", action.size());
    return action;
  }
  
  Eigen::VectorXd reordered(19);
  for(int i = 0; i < 19; i++) {
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
  
  const auto & robot = this->robot();
  
  Eigen::Vector3d baseAngVel = robot.bodyVelW()[0].angular();
  obs.segment(0, 3) = baseAngVel; //base angular vel
  
  Eigen::Matrix3d baseRot = robot.posW().rotation().transpose();
  Eigen::Vector3d rpy = mc_rbdyn::rpyFromMat(baseRot);
  obs(3) = rpy(0);  // roll
  obs(4) = rpy(1);  // pitch
  
  //pos & vel
  Eigen::VectorXd allPos(19), allVel(19);
  for(size_t i = 0; i < allJoints_.size(); ++i)
  {
    if(robot.hasJoint(allJoints_[i]))
    {
      auto jIndex = robot.jointIndexByName(allJoints_[i]);

      allPos(i) = robot.mbc().q[jIndex][0];
      allVel(i) = robot.mbc().alpha[jIndex][0];
    }
    else
    {
      mc_rtc::log::warning("Joint {} not found in robot model", allJoints_[i]);
      allPos(i) = 0.0;
      allVel(i) = 0.0;
    }
  }
  
  Eigen::VectorXd reorderedPos = reorderObservationToManiskill(allPos);
  Eigen::VectorXd reorderedVel = reorderObservationToManiskill(allVel);
  
  std::vector<int> legIndicesInManiskill = {0, 1, 3, 4, 7, 8, 11, 12, 15, 16};
  Eigen::VectorXd legPos(10), legVel(10);
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
  
  // past action
  obs.segment(25, 10) = pastAction_.head(10);
  
  return obs;
}

void RLController::applyAction(const Eigen::VectorXd & action)
{
  if(action.size() != 19)
  {
    mc_rtc::log::error("Action size mismatch: expected 19, got {}", action.size());
    return;
  }
  
  Eigen::VectorXd reorderedAction = reorderActionFromManiskill(action);
  
  pastAction_ = reorderedAction;

  Eigen::VectorXd currentPos(19);
  Eigen::VectorXd currentVel(19);
  
  const auto & robot = this->robot();
  
  for(size_t i = 0; i < allJoints_.size(); ++i)
  {
    if(robot.hasJoint(allJoints_[i]))
    {
      auto jIndex = robot.jointIndexByName(allJoints_[i]);
      
      currentPos(i) = robot.mbc().q[jIndex][0];
      currentVel(i) = robot.mbc().alpha[jIndex][0];
    }
    else
    {
      currentPos(i) = 0.0;
      currentVel(i) = 0.0;
    }
  }
  
  // Eigen::VectorXd desiredTorques = computeImpedanceTorques(reorderedAction, currentPos, currentVel);
  
  std::map<std::string, std::vector<double>> torqueMap;
  for(size_t i = 0; i < 10; ++i)
  {
    if(i >= 10) {
      mc_rtc::log::error("Trying to access legjoint[{}] but size is {}", i, 10);
      break;
    }
    mc_rtc::log::info("Adding pos for joint {} ({}): {}", i, allJoints_[i], reorderedAction(i));
    torqueMap[allJoints_[i]] = {reorderedAction(i)};
  }
  mc_rtc::log::info("========================================================================");
  postureTask_->target(torqueMap);
}

Eigen::VectorXd RLController::computeImpedanceTorques(const Eigen::VectorXd & desiredPos, 
                                                     const Eigen::VectorXd & currentPos,
                                                     const Eigen::VectorXd & currentVel)
{
  // Impedance control: τ = Kp * (q_desired - q_current) + Kd * (0 - dq_current)
  
  Eigen::VectorXd positionError = desiredPos - currentPos;
  Eigen::VectorXd velocityError = -currentVel;  // Desired velocity is 0
  
  Eigen::VectorXd torques = kp_.cwiseProduct(positionError) + kd_.cwiseProduct(velocityError);
  
  return torques;
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