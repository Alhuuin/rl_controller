#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <cstddef>

class PolicySimulatorHandling
{
public:
    PolicySimulatorHandling();
    PolicySimulatorHandling(const std::string& simulator_name, const std::string& robot_name);
    ~PolicySimulatorHandling();
  
    Eigen::VectorXd reorderJointsToSimulator(const Eigen::VectorXd & obs, std::size_t dofNumber);
    Eigen::VectorXd reorderJointsFromSimulator(const Eigen::VectorXd & action, std::size_t dofNumber);
    std::vector<int> invertMapping(const std::vector<int>& jointsMap);
    std::vector<int> getSimulatorIndices(std::vector<int> mcRtcIndices) const;

    std::string simulatorName;
    std::string robotName;
    
private:
    std::vector<int> simuToMcRtcIdx_;    // Joint Mapping simu -> mc_rtc
    std::vector<int> mcRtcToSimuIdx_;    // Joint Mapping mc_rtc -> simu
};

