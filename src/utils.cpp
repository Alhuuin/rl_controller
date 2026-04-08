#include "utils.h"
#include <Eigen/src/Core/Matrix.h>
#include <mc_rtc/logging.h>

#include "RLController.h"

void utils::start_rl_state(mc_control::fsm::Controller & ctl_, std::string state_name)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  state_name_ = state_name;
  mc_rtc::log::info("{} state started", state_name);

  syncTime_ = ctl.policyPeriodMs / 1000;
    
  if(!ctl.rlPolicy || !ctl.rlPolicy->isLoaded())
  {
    mc_rtc::log::error("RL policy not loaded in {} state", state_name);
    return;
  }

  ctl.gui()->addElement(
    {"RLController", state_name},
    mc_rtc::gui::Label("Policy Loaded", [&ctl]() { 
      return ctl.rlPolicy->isLoaded() ? "Yes" : "No"; 
    }),
    mc_rtc::gui::Label("Observation Size", [&ctl]() { 
      return std::to_string(ctl.rlPolicy->getObservationSize()); 
    }),
    mc_rtc::gui::Label("Action Size", [&ctl]() { 
      return std::to_string(ctl.rlPolicy->getActionSize()); 
    })
  );

  mc_rtc::log::success("{} state initialization completed", state_name);
}

void utils::run_rl_state(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  try
  {
    syncTime_ += ctl.timeStep;
    syncPhase_ += ctl.timeStep;
    ctl.phase = fmod(syncPhase_ * ctl.phaseFreq * 2.0 * M_PI, 2.0 * M_PI);
    if(syncTime_ >= ctl.policyPeriodMs/1000)
    {
      // syncTime_ -=ctl.timeStep;
      // // mc_rtc::log::info("FREQ: {:.1f} Hz", 1.0 / (syncTime_));
      ctl.currentObservation = getCurrentObservation(ctl);
      ctl.currentAction = ctl.rlPolicy->predict(ctl.currentObservation);
      applyAction(ctl, ctl.currentAction);
      syncTime_ = 0.0;
    }
  }
  catch(const std::exception & e)
  {
    Eigen::VectorXd zeroAction = Eigen::VectorXd::Zero(ctl.rlPolicy->getActionSize());
    applyAction(ctl, zeroAction);
  }
}

void utils::teardown_rl_state(mc_control::fsm::Controller & ctl_)
{
  ctl_.gui()->removeCategory({"RLController", state_name_});
}

Eigen::VectorXd utils::getCurrentObservation(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  // Observation: [base angular velocity (3), roll (1), pitch (1), joint pos (10), joint vel (10), past action (10), sin(phase) (1), cos(phase) (1), command (3)]

  Eigen::VectorXd obs(ctl.rlPolicy->getObservationSize());
  obs = Eigen::VectorXd::Zero(ctl.rlPolicy->getObservationSize());

  // const auto & robot = this->robot();

  // auto & robot = ctl.robots()[0];
  auto & real_robot = ctl.realRobot(ctl.robots()[0].name());
  auto & imu = ctl.robot().bodySensor("Accelerometer");

  auto q_map = real_robot.encoderValues();
  auto q_dot_map = real_robot.encoderVelocities();

  Eigen::VectorXd q = Eigen::VectorXd::Map(q_map.data(), int(q_map.size()));
  Eigen::VectorXd q_dot = Eigen::VectorXd::Map(q_dot_map.data(), int(q_dot_map.size()));

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

      Eigen::VectorXd reorderedPos = ctl.policySimulatorHandling->reorderJointsToSimulator(q, ctl.dofNumber);
      Eigen::VectorXd reorderedVel = ctl.policySimulatorHandling->reorderJointsToSimulator(q_dot, ctl.dofNumber);

      ctl.legPos_prev_prev = ctl.legPos_prev;
      ctl.legPos_prev = ctl.legPos;
      ctl.legVel_prev_prev = ctl.legVel_prev;
      ctl.legVel_prev = ctl.legVel;
      ctl.legAction_prev_prev = ctl.legAction_prev;
      ctl.legAction_prev = ctl.legAction;

      for(int i = 0; i < int(ctl.usedJoints_simuOrder.size()); ++i)
      {
        int idx = ctl.usedJoints_simuOrder[size_t(i)];
        if(idx >= reorderedPos.size()) {
          mc_rtc::log::error("Leg joint index {} out of bounds for reordered size {}", idx, reorderedPos.size());
          ctl.legPos(i) = 0.0;
          ctl.legVel(i) = 0.0;
        } else {
          ctl.legPos(i) = reorderedPos(idx);
          ctl.legVel(i) = reorderedVel(idx);
        }
        if(idx >= ctl.a_simuOrder.size()) {
          mc_rtc::log::error("Past action index {} out of bounds for size {}", idx, ctl.a_simuOrder.size());
          ctl.legAction(i) = 0.0;
        } else {
          ctl.legAction(i) = ctl.a_simuOrder(idx);
        }
      }

      obs.segment(5, 10) = ctl.legPos;
      obs.segment(15, 10) = ctl.legVel;
      obs.segment(25, 10) = ctl.legAction;
      break;
    }
    case 1:
    {
      ctl.baseAngVel_prev_prev = ctl.baseAngVel_prev;
      ctl.baseAngVel_prev = ctl.baseAngVel;
      ctl.baseAngVel = imu.angularVelocity();

      // Eigen::Matrix3d baseRot = robot.bodyPosW("pelvis").rotation();
      Eigen::Matrix3d baseRot = imu.orientation().toRotationMatrix().normalized();
      ctl.rpy_prev_prev = ctl.rpy_prev;
      ctl.rpy_prev = ctl.rpy;
      ctl.rpy = mc_rbdyn::rpyFromMat(baseRot);

      Eigen::VectorXd reorderedPos = ctl.policySimulatorHandling->reorderJointsToSimulator(q, ctl.dofNumber);
      Eigen::VectorXd reorderedVel = ctl.policySimulatorHandling->reorderJointsToSimulator(q_dot, ctl.dofNumber);

      ctl.legPos_prev_prev = ctl.legPos_prev;
      ctl.legPos_prev = ctl.legPos;
      ctl.legVel_prev_prev = ctl.legVel_prev;
      ctl.legVel_prev = ctl.legVel;
      ctl.legAction_prev_prev = ctl.legAction_prev;
      ctl.legAction_prev = ctl.legAction;

      for(int i = 0; i < int(ctl.usedJoints_simuOrder.size()); ++i)
      {
        int idx = ctl.usedJoints_simuOrder[size_t(i)];
        if(idx >= reorderedPos.size()) {
          mc_rtc::log::error("Leg joint index {} out of bounds for reordered size {}", idx, reorderedPos.size());
          ctl.legPos(i) = 0.0;
          ctl.legVel(i) = 0.0;
        } else {
          ctl.legPos(i) = reorderedPos(idx);
          ctl.legVel(i) = reorderedVel(idx);
        }
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
      obs(105) = cos(ctl.phase);
      obs(106) = sin(ctl.phase);
      obs.segment(107, 3) = ctl.velCmdRL;
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

void utils::applyAction(mc_control::fsm::Controller & ctl_, const Eigen::VectorXd & action)
{
  auto & ctl = static_cast<RLController&>(ctl_);
  Eigen::VectorXd fullAction;

  if (action.size() == ctl.dofNumber)
    fullAction = action;
  else
  {
    // construct the full action vector, setting the unused joints action to 0
    fullAction = Eigen::VectorXd::Zero(ctl.dofNumber);

    for(int i = 0; i < int(ctl.usedJoints_simuOrder.size()); ++i)
    {
      int idx = ctl.usedJoints_simuOrder[size_t(i)];
      if(idx >= fullAction.size()) {
        mc_rtc::log::error("Joint index {} out of bounds for fullAction size {}", idx, fullAction.size());
      } else if(static_cast<Eigen::Index>(i) >= action.size()) {
        mc_rtc::log::error("Action index {} out of bounds for action size {}", i, action.size());
      } else {
        fullAction(idx) = action(i); // Set leg joint action
      }
    }
  }

  // Get current observation for logging
  Eigen::VectorXd currentObs = getCurrentObservation(ctl);
  
  // Run new inference and update target position, scaled by action scale
  ctl.a_vector = ctl.policySimulatorHandling->reorderJointsFromSimulator(fullAction, ctl.dofNumber) * ctl.actionScale;
  ctl.q_rl = ctl.q_zero_vector + ctl.a_vector;

  // For not controlled joints, use the zero position
  for (int i = 0; i < ctl.dofNumber; ++i)
  {
    auto it = std::find(ctl.usedJoints_mcRtcOrder.begin(), ctl.usedJoints_mcRtcOrder.end(), i);
    if(it == ctl.usedJoints_mcRtcOrder.end())
    {
      ctl.q_rl(i) = ctl.q_zero_vector(i); // Set to zero position
    }
  }

  ctl.a_simuOrder = ctl.policySimulatorHandling->reorderJointsToSimulator(ctl.a_vector, ctl.dofNumber);
}
