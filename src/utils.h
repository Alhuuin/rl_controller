#pragma once
#include <mc_control/fsm/State.h>
#include <thread>
#include <condition_variable>

struct utils
{  
    // RL states
    void start_rl_state(mc_control::fsm::Controller & ctl_, std::string state_name);
    void run_rl_state(mc_control::fsm::Controller & ctl_, std::string state_name);
    void teardown_rl_state(mc_control::fsm::Controller & ctl_, std::string state_name);

    // mc_rtc - RL policy interface
    Eigen::VectorXd getCurrentObservation(mc_control::fsm::Controller & ctl_);
    /* Return true if a newAction was applied */
    bool applyAction(mc_control::fsm::Controller & ctl_, const Eigen::VectorXd & action);

    // Threading
    void startInferenceThread(mc_control::fsm::Controller & ctl_);
    void stopInferenceThread();
    void inferenceThreadFunction(mc_control::fsm::Controller & ctl_);
    void updateObservationForInference(mc_control::fsm::Controller & ctl_);
    Eigen::VectorXd getLatestAction(mc_control::fsm::Controller & ctl_);

    // threading
    std::unique_ptr<std::thread> inferenceThread_;
    std::mutex actionMutex_;
    std::mutex observationMutex_;
    std::condition_variable inferenceCondition_;
    Eigen::VectorXd action;

    private:
        // State-specific data
        size_t stepCount_ = 0;
        double startTime_ = 0.0;
        double syncTime_;
        double syncPhase_ = 0.0;
        std::atomic<bool> shouldStopInference_ = false;
        std::atomic<bool> newObservationAvailable_ = false;
        std::atomic<bool> newActionAvailable_ = false;
        std::chrono::steady_clock::time_point lastInferenceTime_;
        bool shouldRunInference_ = true;
};