#include "utils.h"
#include <cstddef>
#include <mc_rtc/logging.h>

#include "RLController.h"

void utils::start_rl_state(mc_control::fsm::Controller & ctl_, std::string state_name)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  mc_rtc::log::info("{} state started", state_name);
  lastInferenceTime_ = std::chrono::steady_clock::now();
  action = Eigen::VectorXd::Zero(ctl.rlPolicy_->getActionSize());

  stepCount_ = 0;
  syncTime_ = INFERENCE_PERIOD_MS / 1000;
  startTime_ = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
  if(!ctl.rlPolicy_ || !ctl.rlPolicy_->isLoaded())
  {
    mc_rtc::log::error("RL policy not loaded in {} state", state_name);
    return;
  }

  mc_rtc::log::success("{} state initialization completed", state_name);

  ctl.gui()->addElement(
    {"RLController", state_name},
    mc_rtc::gui::Label("Steps", [this]() { return std::to_string(stepCount_); }),
    mc_rtc::gui::Label("Policy Loaded", [&ctl]() { 
      return ctl.rlPolicy_->isLoaded() ? "Yes" : "No"; 
    }),
    mc_rtc::gui::Label("Observation Size", [&ctl]() { 
      return std::to_string(ctl.rlPolicy_->getObservationSize()); 
    }),
    mc_rtc::gui::Label("Action Size", [&ctl]() { 
      return std::to_string(ctl.rlPolicy_->getActionSize()); 
    })
  );
}

void utils::run_rl_state(mc_control::fsm::Controller & ctl_, std::string state_name)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  
  auto startTime = std::chrono::high_resolution_clock::now();
  
  try
  {
    if(ctl.useAsyncInference_)
    {
      updateObservationForInference(ctl);
      action = getLatestAction(ctl);
      // Apply a new action and log if a new action was applied
      bool newActionApplied = applyAction(ctl, action);
      if(newActionApplied)
      {
        stepCount_++;
        if(ctl.logTiming_ && (stepCount_ % ctl.timingLogInterval_ == 0))
        {
          auto endTime = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
          
          double currentTime = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
          double avgFreq = stepCount_ / (currentTime - startTime_);
          
          const char* mode = ctl.useAsyncInference_ ? "async" : "sync";
          mc_rtc::log::info("{} Step {} ({}): inference time = {} μs, avg policy freq = {:.1f} Hz",
                            state_name, stepCount_, mode, duration.count(), avgFreq);
        }
      }    
    }
    else if(syncTime_ >= INFERENCE_PERIOD_MS/1000)
    {
      // mc_rtc::log::info("FREQ: {:.1f} Hz", 1.0 / (syncTime_));
      syncPhase_ += ctl.timeStep;
      ctl.phase_ = fmod(syncPhase_ * ctl.phaseFreq_ * 2.0 * M_PI, 2.0 * M_PI);
      ctl.currentObservation_ = getCurrentObservation(ctl);
      ctl.currentAction_ = ctl.rlPolicy_->predict(ctl.currentObservation_);
      applyAction(ctl, ctl.currentAction_);
      syncTime_ = 0.0;
    }
    else
    {
      syncTime_ += ctl.timeStep;
      syncPhase_ += ctl.timeStep;
      ctl.phase_ = fmod(syncPhase_ * ctl.phaseFreq_ * 2.0 * M_PI, 2.0 * M_PI);
    }
  }
  catch(const std::exception & e)
  {
    mc_rtc::log::error("{} error at step {}: {}", state_name, stepCount_, e.what());

    Eigen::VectorXd zeroAction = Eigen::VectorXd::Zero(ctl.rlPolicy_->getActionSize());
    applyAction(ctl, zeroAction);
  }
}

void utils::teardown_rl_state(mc_control::fsm::Controller & ctl_, std::string state_name)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  mc_rtc::log::info("{} state ending after {} steps", state_name, stepCount_);

  ctl_.gui()->removeCategory({"RLController", state_name});
  
  double currentTime = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  double totalTime = currentTime - startTime_;
  double avgFreq = stepCount_ / totalTime;

  mc_rtc::log::info("{} final stats: {} steps in {:.2f}s, avg freq = {:.1f} Hz",
                    state_name, stepCount_, totalTime, avgFreq);
}

Eigen::VectorXd utils::getCurrentObservation(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  // Observation: [base angular velocity (3), roll (1), pitch (1), joint pos (10), joint vel (10), past action (10), sin(phase) (1), cos(phase) (1), command (3)]

  Eigen::VectorXd obs(ctl.rlPolicy_->getObservationSize());
  obs = Eigen::VectorXd::Zero(ctl.rlPolicy_->getObservationSize());

  // const auto & robot = this->robot();

  auto & robot = ctl.robots()[0];
  auto & real_robot = ctl.realRobot(ctl.robots()[0].name());
  auto & imu = ctl.robot().bodySensor("Accelerometer");
  
  // ctl.baseAngVel = robot.bodyVelW("pelvis").angular();
  ctl.baseAngVel_prev_prev = ctl.baseAngVel_prev;
  ctl.baseAngVel_prev = ctl.baseAngVel;
  ctl.baseAngVel = imu.angularVelocity();
  obs(0) = ctl.baseAngVel.x(); //base angular vel
  obs(1) = ctl.baseAngVel.y();
  obs(2) = ctl.baseAngVel.z();

  // Eigen::Matrix3d baseRot = robot.bodyPosW("pelvis").rotation();
  Eigen::Matrix3d baseRot = imu.orientation().toRotationMatrix().normalized();
  ctl.rpy_prev_prev = ctl.rpy_prev;
  ctl.rpy_prev = ctl.rpy;
  ctl.rpy = mc_rbdyn::rpyFromMat(baseRot);
  obs(3) = ctl.rpy(0);  // roll
  obs(4) = ctl.rpy(1);  // pitch

  Eigen::VectorXd reorderedPos = ctl.policySimulatorHandling_->reorderJointsToSimulator(ctl.currentPos, ctl.dofNumber);
  Eigen::VectorXd reorderedVel = ctl.policySimulatorHandling_->reorderJointsToSimulator(ctl.currentVel, ctl.dofNumber);

  ctl.legPos_prev_prev = ctl.legPos_prev;
  ctl.legPos_prev = ctl.legPos;
  ctl.legVel_prev_prev = ctl.legVel_prev;
  ctl.legVel_prev = ctl.legVel;
  ctl.legAction_prev_prev = ctl.legAction_prev;
  ctl.legAction_prev = ctl.legAction;

  for(size_t i = 0; i < ctl.usedJoints_simuOrder.size(); ++i)
  {
    int idx = ctl.usedJoints_simuOrder[i];
    if(idx >= reorderedPos.size()) {
      mc_rtc::log::error("Leg joint index {} out of bounds for reordered size {}", idx, reorderedPos.size());
      ctl.legPos(i) = 0.0;
      ctl.legVel(i) = 0.0;
    } else {
      ctl.legPos(i) = reorderedPos(idx);
      ctl.legVel(i) = reorderedVel(idx);
    }
  }

  obs.segment(5, 10) = ctl.legPos;
  obs.segment(15, 10) = ctl.legVel;

  // past action: reorder to Simulator format and extract leg joints
  for(size_t i = 0; i < ctl.usedJoints_simuOrder.size(); ++i)
  {
    int idx = ctl.usedJoints_simuOrder[i];
    if(idx >= ctl.a_simuOrder.size()) {
      mc_rtc::log::error("Past action index {} out of bounds for size {}", idx, ctl.a_simuOrder.size());
      ctl.legAction(i) = 0.0;
    } else {
      ctl.legAction(i) = ctl.a_simuOrder(idx);
    }
  }
  obs.segment(25, 10) = ctl.legAction;

  // Addition for walking policy:
  // Phase
  if(ctl.useAsyncInference_)
  {
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - ctl.startPhase_);
    ctl.phase_ = fmod(elapsed.count() * 0.001 * ctl.phaseFreq_ * 2.0 * M_PI, 2.0 * M_PI);
  }

  obs(35) = sin(ctl.phase_);
  obs(36) = cos(ctl.phase_);

  // Command (3 elements) - [vx, vy, yaw_rate]
  obs.segment(37, 3) = ctl.velCmdRL_;

  obs.segment(0, 3) = ctl.baseAngVel * 0.25;
  obs.segment(3, 3) = ctl.baseAngVel_prev * 0.25;
  obs.segment(6, 3) = ctl.baseAngVel_prev_prev * 0.25;
  obs.segment(9, 2) = ctl.rpy.segment(0,2);
  obs.segment(11, 2) = ctl.rpy_prev.segment(0,2);
  obs.segment(13, 2) = ctl.rpy_prev_prev.segment(0,2);
  obs.segment(15, 10) = ctl.legPos;
  obs.segment(25, 10) = ctl.legPos_prev;
  obs.segment(35, 10) = ctl.legPos_prev_prev;
  obs.segment(45, 10) = ctl.legVel* 0.05;
  obs.segment(55, 10) = ctl.legVel_prev* 0.05;
  obs.segment(65, 10) = ctl.legVel_prev_prev* 0.05;
  obs.segment(75, 10) = ctl.legAction;
  obs.segment(85, 10) = ctl.legAction_prev;
  obs.segment(95, 10) = ctl.legAction_prev_prev;
  obs(105) = cos(ctl.phase_);
  obs(106) = sin(ctl.phase_);
  obs.segment(107, 3) = ctl.velCmdRL_;
  
  return obs;
}

bool utils::applyAction(mc_control::fsm::Controller & ctl_, const Eigen::VectorXd & action)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  bool newActionApplied = false;

  Eigen::VectorXd fullAction = Eigen::VectorXd::Zero(ctl.dofNumber);
  
  // In that policy action vector contains only leg joints (10)
  // Construct full action vector with zeros for non-leg joints
  for(size_t i = 0; i < ctl.usedJoints_simuOrder.size(); ++i)
  {
    int idx = ctl.usedJoints_simuOrder[i];
    if(idx >= fullAction.size()) {
      mc_rtc::log::error("Leg joint index {} out of bounds for fullAction size {}", idx, fullAction.size());
    } else if(i >= action.size()) {
      mc_rtc::log::error("Action index {} out of bounds for action size {}", i, action.size());
    } else {
      fullAction(idx) = action(i); // Set leg joint action
    }
  }

  if(shouldRunInference_) {
    newActionApplied = true;
    // Get current observation for logging
    Eigen::VectorXd currentObs = getCurrentObservation(ctl);
    
    // Update lastActions_
    ctl.a_before_vector = ctl.a_vector;
    // Run new inference and update target position
    ctl.a_vector = ctl.policySimulatorHandling_->reorderJointsFromSimulator(fullAction, ctl.dofNumber);
    ctl.q_rl = ctl.q_zero_vector + ctl.a_vector;

    // For not controlled joints, use the zero position
    for (size_t i = 0; i < ctl.dofNumber; ++i)
    {
      auto it = std::find(ctl.usedJoints_mcRtcOrder.begin(), ctl.usedJoints_mcRtcOrder.end(), i);
      if(it == ctl.usedJoints_mcRtcOrder.end())
      {
        ctl.q_rl(i) = ctl.q_zero_vector(i); // Set to zero position
      }
    }

    ctl.a_simuOrder = ctl.policySimulatorHandling_->reorderJointsToSimulator(ctl.a_vector, ctl.dofNumber);

    static int inferenceCounter = 0;
    inferenceCounter++;
  }
  
  // Get current joint positions and velocities
  Eigen::VectorXd q_current(ctl.dofNumber);
  Eigen::VectorXd q_dot_current(ctl.dofNumber);
  auto & real_robot = ctl.realRobot(ctl.robots()[0].name());
  auto q = real_robot.encoderValues();
  q_current = Eigen::VectorXd::Map(q.data(), q.size());
  auto vel = real_robot.encoderVelocities();
  q_dot_current = Eigen::VectorXd::Map(vel.data(), vel.size());

  auto & robot = ctl.robots()[0];;
  
  for(size_t i = 0; i < ctl.mcRtcJointsOrder.size(); ++i)
  {
    if(robot.hasJoint(ctl.mcRtcJointsOrder[i]))
    {
      auto jIndex = robot.jointIndexByName(ctl.mcRtcJointsOrder[i]);
      q_current(i) = robot.mbc().q[jIndex][0];
      q_dot_current(i) = robot.mbc().alpha[jIndex][0];
    }
    else
    {
      q_current(i) = 0.0;
      q_dot_current(i) = 0.0;
    }
  }
  return newActionApplied;
}

void utils::startInferenceThread(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  mc_rtc::log::info("Starting RL inference thread");
  inferenceThread_ = std::make_unique<std::thread>([this, &ctl](){
        inferenceThreadFunction(ctl);
    });
}

void utils::stopInferenceThread()
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

void utils::inferenceThreadFunction(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
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
    Eigen::VectorXd obs = ctl.currentObservation_;
    newObservationAvailable_ = false;
    lock.unlock();
    
    try
    {
      // Check if it's time for new inference (40Hz = 25ms period)
      auto currentTime = std::chrono::steady_clock::now();
      auto timeSinceLastInference = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastInferenceTime_);
      // auto startTime = std::chrono::high_resolution_clock::now();
      shouldRunInference_ = timeSinceLastInference.count() >= INFERENCE_PERIOD_MS;

      newActionAvailable_ = shouldRunInference_;

      if(shouldRunInference_)
      {
        Eigen::VectorXd action = ctl.rlPolicy_->predict(obs);
        lastInferenceTime_ = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> actionLock(actionMutex_);
        ctl.currentAction_ = action;
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

void utils::updateObservationForInference(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  Eigen::VectorXd obs = getCurrentObservation(ctl);
  
  //update shared observation
  {
    std::lock_guard<std::mutex> lock(observationMutex_);
    ctl.currentObservation_ = obs;
    newObservationAvailable_ = true;
  }
  
  // notify inference thread
  inferenceCondition_.notify_one();
}

Eigen::VectorXd utils::getLatestAction(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  if(newActionAvailable_)
  {
    std::lock_guard<std::mutex> lock(actionMutex_);
    if(newActionAvailable_)
    {
      ctl.latestAction_ = ctl.currentAction_;
      newActionAvailable_ = false;
    }
  }
  return ctl.latestAction_;
} 

