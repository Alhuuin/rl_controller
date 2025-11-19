#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <cstddef>

class PolicySimulatorHandling
{
public:
    PolicySimulatorHandling();
    PolicySimulatorHandling(const std::string& simulator_name);
    ~PolicySimulatorHandling();
  
    Eigen::VectorXd reorderJointsToManiskill(const Eigen::VectorXd & obs, std::size_t dofNumber);
    Eigen::VectorXd reorderJointsFromManiskill(const Eigen::VectorXd & action, std::size_t dofNumber);
    Eigen::VectorXd reorderJointsToSimulator(const Eigen::VectorXd & obs, std::size_t dofNumber);
    Eigen::VectorXd reorderJointsFromSimulator(const Eigen::VectorXd & action, std::size_t dofNumber);
    std::vector<int> getSimulatorIndices(std::vector<int> mcRtcIndices) const;

    std::string simulatorName;
    
    // Maniskill handling
    std::vector<int> maniskillToMcRtcIdx_;    // Action reordering: Maniskill -> mc_rtc
    std::vector<int> mcRtcToManiskillIdx_;    // Observation reordering: mc_rtc -> Maniskill
};

