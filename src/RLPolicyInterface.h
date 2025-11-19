#pragma once

#include <Eigen/Dense>
#include <string>
#include <memory>

// #include <onnxruntime_cxx_api.h>
#include "onnxruntime/include/onnxruntime_cxx_api.h"

/**
 * @brief Interface for RL policy inference
 * 
 * This class abstracts the RL policy implementation, allowing you to use
 * different backends (PyTorch, ONNX, TensorFlow, etc.) without changing
 * the controller code. (currently only ONNX supported)
 */
class RLPolicyInterface
{
public:
  /**
   * @brief Construct with a policy file path
   * @param policyPath Path to the policy file
   */
  RLPolicyInterface(const std::string & policyPath);
  
  /**
   * @brief Destructor
   */
  ~RLPolicyInterface();
  
  /**
   * @brief Run inference on the policy
   * @param observation Input observation vector
   * @return Action vector
   */
  Eigen::VectorXd predict(const Eigen::VectorXd & observation);
  
  /**
   * @brief Check if policy is loaded successfully
   * @return true if policy is ready for inference
   */
  bool isLoaded() const { return isLoaded_; }
  
  /**
   * @brief Get the expected observation size
   * @return Expected observation dimension
   */
  int getObservationSize() const { return inputSize_; }
  
  /**
   * @brief Get the action size
   * @return Action dimension  
   */
  int getActionSize() const { return outputSize_; }

private:
  int inputSize_;
  int outputSize_;
  bool inputIs2D_;    // true if input shape is [batch, obs_size], false if [obs_size]
  bool outputIs2D_;   // true if output shape is [batch, action_size], false if [action_size]
  
  bool isLoaded_;
  std::string policyPath_;
  
  std::unique_ptr<Ort::Session> onnxSession_;
  std::unique_ptr<Ort::Env> onnxEnv_;
  std::unique_ptr<Ort::MemoryInfo> memoryInfo_;
  std::string inputName_;
  std::string outputName_;
  const char* inputNamePtr_;   // Cached pointer for inference performance
  const char* outputNamePtr_;  // Cached pointer for inference performance
  
  void loadPolicy(const std::string & path);
  
  Eigen::VectorXd runOnnxInference(const Eigen::VectorXd & observation);
}; 