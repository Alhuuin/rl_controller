#include "utils.h"
#include <Eigen/src/Core/Matrix.h>
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
  syncTime_ = ctl.policyPeriodMs / 1000;
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
    syncTime_ += ctl.timeStep;
    syncPhase_ += ctl.timeStep;
    ctl.phase_ = fmod(syncPhase_ * ctl.phaseFreq_ * 2.0 * M_PI, 2.0 * M_PI);
    if(syncTime_ >= ctl.policyPeriodMs/1000)
    {
      ctl.currentObservation_ = getCurrentObservation(ctl);
      ctl.currentAction_ = ctl.rlPolicy_->predict(ctl.currentObservation_);
      applyAction(ctl, ctl.currentAction_);
      syncTime_ = 0.0;
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
  mc_rtc::log::info("{} state ending after {} steps", state_name, stepCount_);

  ctl_.gui()->removeCategory({"RLController", state_name});
  
  double currentTime = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  double totalTime = currentTime - startTime_;
  double avgFreq = static_cast<double>(stepCount_) / totalTime;

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

  switch (ctl.currentPolicyIndex) {
    case 0:
    {
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
      break;
    }
    case 1:
    {
      ctl.baseAngVel_prev_prev = ctl.baseAngVel_prev;
      ctl.baseAngVel_prev = ctl.baseAngVel;
      ctl.baseAngVel = imu.angularVelocity();
      obs(0) = ctl.baseAngVel.x(); //base angular vel
      obs(1) = ctl.baseAngVel.y();
      obs(2) = ctl.baseAngVel.z();

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

      // Phase
      obs(35) = sin(ctl.phase_);
      obs(36) = cos(ctl.phase_);

      // Command (3 elements) - [vx, vy, yaw_rate]
      obs.segment(37, 3) = ctl.velCmdRL_;
      break;
    }
    case 2:
    {
      ctl.baseAngVel_prev_prev = ctl.baseAngVel_prev;
      ctl.baseAngVel_prev = ctl.baseAngVel;
      ctl.baseAngVel = imu.angularVelocity();

      // Eigen::Matrix3d baseRot = robot.bodyPosW("pelvis").rotation();
      Eigen::Matrix3d baseRot = imu.orientation().toRotationMatrix().normalized();
      ctl.rpy_prev_prev = ctl.rpy_prev;
      ctl.rpy_prev = ctl.rpy;
      ctl.rpy = mc_rbdyn::rpyFromMat(baseRot);

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
      break;
    }
    case 3:
    {
      const auto & floatingBaseBody = robot.mb().body(0).name();
      Eigen::Vector3d baseLinVelW = robot.bodyVelW(floatingBaseBody).linear();
      Eigen::Vector3d baseAngVelW = robot.bodyVelW(floatingBaseBody).angular();

      Eigen::Vector3d gravity(0, 0, -9.81);
      Eigen::Quaterniond q_imu_to_world;
      auto qInRL = real_robot.mbc().q;
      Eigen::VectorXd floatingBase_qInRL = rbd::paramToVector(real_robot.mb(), qInRL);
      Eigen::VectorXd q_imu_vector = floatingBase_qInRL.segment(0, 4);
      q_imu_to_world.w() = q_imu_vector(0);
      q_imu_to_world.x() = q_imu_vector(1);
      q_imu_to_world.y() = q_imu_vector(2);
      q_imu_to_world.z() = q_imu_vector(3);
      Eigen::Matrix3d R_imu_to_world = q_imu_to_world.toRotationMatrix();

      // velocities expressed in the base frame.
      ctl.baseLinVel = R_imu_to_world.transpose() * baseLinVelW;
      obs(0) = ctl.baseLinVel.x();
      obs(1) = ctl.baseLinVel.y();
      obs(2) = ctl.baseLinVel.z();

      ctl.baseAngVel = R_imu_to_world.transpose() * baseAngVelW;
      obs(3) = ctl.baseAngVel.x();
      obs(4) = ctl.baseAngVel.y();
      obs(5) = ctl.baseAngVel.z();

      Eigen::Vector3d gravity_b = R_imu_to_world.transpose() * gravity;
      Eigen::Vector3d gravity_b_dir = gravity_b.normalized();
      ctl.projected_gravity = gravity_b_dir;
      obs(6) = ctl.projected_gravity.x(); // base linear acc
      obs(7) = ctl.projected_gravity.y();
      obs(8) = ctl.projected_gravity.z();

      obs.segment(9, 3) = ctl.velCmdRL_;


      Eigen::VectorXd reorderedPos = ctl.policySimulatorHandling_->reorderJointsToSimulator(ctl.currentPos, ctl.dofNumber);
      Eigen::VectorXd reorderedVel = ctl.policySimulatorHandling_->reorderJointsToSimulator(ctl.currentVel, ctl.dofNumber);
      Eigen::VectorXd reorderedQZero = ctl.policySimulatorHandling_->reorderJointsToSimulator(ctl.q_zero_vector, ctl.dofNumber);
      for(size_t i = 0; i < ctl.usedJoints_simuOrder.size(); ++i)
      {
        int idx = ctl.usedJoints_simuOrder[i];
        if(idx >= reorderedPos.size()) {
          mc_rtc::log::error("Leg joint index {} out of bounds for reordered size {}", idx, reorderedPos.size());
          ctl.legPos(i) = 0.0;
          ctl.legVel(i) = 0.0;
        } else {
          // IsaacLab observation uses joint_pos_rel = joint_pos - default_joint_pos.
          ctl.legPos(i) = reorderedPos(idx) - reorderedQZero(idx);
          ctl.legVel(i) = reorderedVel(idx);
        }
      }
      obs.segment(12, 12) = ctl.legPos;
      obs.segment(24, 12) = ctl.legVel;

      for(size_t i = 0; i < ctl.usedJoints_simuOrder.size(); ++i)
      {
        int idx = ctl.usedJoints_simuOrder[i];
        if(idx >= ctl.a_simuOrder.size()) {
          mc_rtc::log::error("Past action index {} out of bounds for size {}", idx, ctl.a_simuOrder.size());
          ctl.legAction(i) = 0.0;
        } else {
          // here storing raw policy outputs, not scaled joint deltas
          ctl.legAction(i) = (std::abs(ctl.actionScale) > 1e-9) ? ctl.a_simuOrder(idx) / ctl.actionScale : 0.0;
        }
      }
      obs.segment(36, 12) = ctl.legAction;
      break;
    }
    case 4 :
    {
      Eigen::Matrix3d baseRot = imu.orientation().toRotationMatrix().normalized();
      ctl.rpy_prev_prev = ctl.rpy_prev;
      ctl.rpy_prev = ctl.rpy;
      ctl.rpy = mc_rbdyn::rpyFromMat(baseRot);
      // root_r 1
      obs(0) = ctl.rpy(0);
      // root_p 1
      obs(1) = ctl.rpy(1);
      // root_ang_vel 3
      ctl.baseAngVel = imu.angularVelocity();
      obs(2) = ctl.baseAngVel.x();
      obs(3) = ctl.baseAngVel.y();
      obs(4) = ctl.baseAngVel.z();


      Eigen::VectorXd reorderedPos = ctl.policySimulatorHandling_->reorderJointsToSimulator(ctl.currentPos, ctl.dofNumber);
      Eigen::VectorXd reorderedVel = ctl.policySimulatorHandling_->reorderJointsToSimulator(ctl.currentVel, ctl.dofNumber);

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
      // motor_pos 12
      obs.segment(5, 12) = ctl.legPos;
      // motor_vel 12
      obs.segment(17, 12) = ctl.legVel;
      // leg_ctrl 12 ???
      obs.segment(29, 12) = ctl.legAction;
      // clock 2
      obs(41) = sin(ctl.phase_);
      obs(42) = cos(ctl.phase_);
      // mode 3
      obs.segment(43,3) = Eigen::Vector3d(0,1,0);
      // ref_mode 3
      obs.segment(46,3) = Eigen::Vector3d(0,1,0);
      for (int i = 0; i<49;i++)
      {
        mc_rtc::log::info("index {}: {}", i, obs[i]);
      }
      break;
    }
    default:
    {
      mc_rtc::log::error("Unknown policy index: {}", ctl.currentPolicyIndex);
      break;
    }
  }
  
  return obs;
}

bool utils::applyAction(mc_control::fsm::Controller & ctl_, const Eigen::VectorXd & action)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  bool newActionApplied = false;
  Eigen::VectorXd fullAction;

  if (static_cast<size_t>(action.size()) == ctl.dofNumber)
    fullAction = action;
  else
  {
    // construct the full action vector, setting the unused joints action to 0
    fullAction = Eigen::VectorXd::Zero(ctl.dofNumber);

    for(size_t i = 0; i < ctl.usedJoints_simuOrder.size(); ++i)
    {
      int idx = ctl.usedJoints_simuOrder[i];
      if(idx >= fullAction.size()) {
        mc_rtc::log::error("Joint index {} out of bounds for fullAction size {}", idx, fullAction.size());
      } else if(static_cast<Eigen::Index>(i) >= action.size()) {
        mc_rtc::log::error("Action index {} out of bounds for action size {}", i, action.size());
      } else {
        fullAction(idx) = action(i); // Set leg joint action
      }
    }
  }

  if(shouldRunInference_) {
    newActionApplied = true;
    // Get current observation for logging
    Eigen::VectorXd currentObs = getCurrentObservation(ctl);
    
    // Update lastActions_
    ctl.a_before_vector = ctl.a_vector;
    // Run new inference and update target position, scaled by action scale
    ctl.a_vector = ctl.policySimulatorHandling_->reorderJointsFromSimulator(fullAction, ctl.dofNumber) * ctl.actionScale;
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