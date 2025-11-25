#include "PolicySimulatorHandling.h"
#include <mc_rtc/logging.h>
#include <string>
#include <vector>

PolicySimulatorHandling::PolicySimulatorHandling()
{
  // Default constructor - no simulator handling
}

PolicySimulatorHandling::PolicySimulatorHandling(const std::string& simulator_name, const std::string& robot_name):
    simulatorName(simulator_name), robotName(robot_name)
{
  // check if simuToMcRtcIdx_ needs initialization
  if (!simuToMcRtcIdx_.empty())
  {
    mc_rtc::log::warning("Simulator mapping for {} on {} manually initialized in header", simulator_name, robot_name);
  }
  if(robot_name == "H1" && (simulator_name == "Maniskill" || simulator_name == "IsaacLab"))
  {
      mcRtcToSimuIdx_ = {0, 5, 10, 1, 6, 11, 15, 2, 7, 12, 16, 3, 8, 13, 17, 4, 9, 14, 18};
      simuToMcRtcIdx_ = invertMapping(mcRtcToSimuIdx_);
  }
  else {
      mc_rtc::log::error_and_throw("Unsupported simulator or robot: {} with {}, please specify a mc_rtc to simulator joint order mapping in PolicySimulatorHandling.h", simulator_name, robot_name);
  }
}

PolicySimulatorHandling::~PolicySimulatorHandling()
{
}

Eigen::VectorXd PolicySimulatorHandling::reorderJointsToSimulator(const Eigen::VectorXd & obs, size_t dofNumber)
{  
  if(obs.size() != dofNumber) {
    mc_rtc::log::error("Observation reordering expects dofNumber joints, got {}", obs.size());
    return obs;
  }
  
  Eigen::VectorXd reordered = Eigen::VectorXd::Zero(dofNumber);
  for(int i = 0; i < dofNumber; i++) {
    if(i >= mcRtcToSimuIdx_.size()) {
      mc_rtc::log::error("Trying to access mcRtcToSimuIdx_[{}] but size is {}", i, mcRtcToSimuIdx_.size());
      reordered[i] = 0.0;
      continue;
    }
    
    int srcIdx = mcRtcToSimuIdx_[i];
    if(srcIdx >= obs.size()) {
      mc_rtc::log::error("Index {} out of bounds for obs size {}", srcIdx, obs.size());
      reordered[i] = 0.0;
    }
    else {
      reordered[i] = obs(srcIdx);
    }
  }
  return reordered;
}

Eigen::VectorXd PolicySimulatorHandling::reorderJointsFromSimulator(const Eigen::VectorXd & action, size_t dofNumber)
{  
  if(action.size() != dofNumber) {
    mc_rtc::log::error("Action reordering expects dofNumber joints, got {}", action.size());
    return action;
  }
  
  Eigen::VectorXd reordered = Eigen::VectorXd::Zero(dofNumber);
  for(int i = 0; i < dofNumber; i++) {
    if(i >= simuToMcRtcIdx_.size()) {
      mc_rtc::log::error("Trying to access simuToMcRtcIdx_[{}] but size is {}", i, simuToMcRtcIdx_.size());
      reordered[i] = 0.0;
      continue;
    }
    
    int srcIdx = simuToMcRtcIdx_[i];
    if(srcIdx >= action.size()) {
      mc_rtc::log::error("Action reorder index {} out of bounds for action size {}", srcIdx, action.size());
      reordered[i] = 0.0;
    }
    else {
      reordered[i] = action(srcIdx);
    }
  }
  return reordered;
}

std::vector<int> PolicySimulatorHandling::invertMapping(const std::vector<int>& jointsMap)
{
  std::vector<int> simuToMcRtc(jointsMap.size(), -1);

  for (size_t i = 0; i < jointsMap.size(); ++i)
  {
      int simu = jointsMap[i];
      simuToMcRtc[simu] = i;
  }

  return simuToMcRtc;
}

std::vector<int> PolicySimulatorHandling::getSimulatorIndices(std::vector<int> mcRtcIndices) const
{
  std::vector<int> simuIndices;
  for(int idx : mcRtcIndices)
  {
      if(idx < simuToMcRtcIdx_.size())
      {
          simuIndices.push_back(simuToMcRtcIdx_[idx]);
      }
      else
      {
          mc_rtc::log::error("Index {} out of bounds for simuToMcRtcIdx_ size {}", idx, simuToMcRtcIdx_.size());
      }
  }
  return simuIndices;
}