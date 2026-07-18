# RLController

Acc-CBF-QP was introduced in:

> **Safe Execution of RL Policies via Acceleration-based CBF-QP Constraint Enforcement for Real-World Robotic Deployments**
> Bastien Muraccioli, Alice Cariou, Pierre-Alexandre Leziart, Mathieu Celerier, Arnaud Demont, Gentiane Venture, Mehdi Benallegue
> IROS 2026 — [Paper](https://hal.science/hal-05362571) · [Project page](https://safe-rl-qp.github.io/)

Part of the Acc-CBF-QP ecosystem: [paper implementation](https://github.com/safe-rl-qp/mc-safe-rl-qp) · [superbuild](https://github.com/safe-rl-qp/safe-rl-qp-mc-rtc-superbuild) · [controller template](https://github.com/bastien-muraccioli/new-rl-qp-controller) · [community controllers](https://github.com/safe-rl-qp/awesome-safe-rl-qp)

An FSM controller that integrates reinforcement learning policies with [mc_rtc](https://jrl-umi3218.github.io/mc_rtc/) for robotic control. This package provides example policies for the H1 humanoid robot. It currently only supports ONNX format for policy deployment.

**Note**: ONNX Runtime is bundled with this repository—no external installation required.

## Architecture

The controller is organized into the following components:

- **[etc/RLController.in.yaml](etc/RLController.in.yaml)**: Main configuration file for the controller
- **[RLController](src/RLController.cpp)**: Core FSM controller that integrates RL policies with mc_rtc
- **[RLPolicyInterface](src/RLPolicyInterface.cpp)**: Handles ONNX model loading and inference
- **[PolicySimulatorHandling](src/PolicySimulatorHandling.cpp)**: Manages different rl training environment (e.g., joint ordering differences)
- **[states/RL_State](src/states/RL_State.cpp)**: FSM state that executes the RL policy and applies torque commands

## Building

The easiest way to get this controller running is through the [safe-rl-qp-mc-rtc-superbuild](https://github.com/safe-rl-qp/safe-rl-qp-mc-rtc-superbuild), which installs mc_rtc, all required dependencies, and this controller in one go — enabling the `WITH_H1` option builds and installs RLController automatically alongside the H1 robot module. Follow the superbuild's README for the full walkthrough; you don't need to follow the manual steps below unless you're building this repo standalone (e.g. while developing it directly, or adapting it via the [controller template](https://github.com/bastien-muraccioli/new-rl-qp-controller)).

## Usage

### Robot and RL Training Environment Support

The controller is optimized for the H1 humanoid robot with minimal configuration required. Support for other robots is possible with additional adaptation (see [Adding a New Robot](#adding-a-new-robot)).

Policies trained in ManiSkill and IsaacLab are fully supported. For policies from other training environments, you can add custom simulator support (see [Adding a New RL Training Environment](#adding-a-new-rl-training-environment)).

### Policy Management

Default policies are located in the [`policy/`](policy/) directory. The controller supports switching between multiple policies at runtime through the GUI (`RLController/Policy` section).

**Important**: Policy transitions should be compatible with the current state. For example, switching from standing to walking works because the walking policy can handle observations from a standing state, but the reverse may not be true without proper handling.

### Velocity Control

For policies that support velocity commands, three control methods are available:

- **Gamepad control**: If a compatible gamepad is plugged into the PC, the controller will automatically use it for velocity commands. The controls are:
  - **Left joystick** or **D-pad**: control the commanded **X** and **Y** linear velocities.
  - **Right joystick (horizontal axis)**: control the commanded **yaw** velocity.

  This functionality is provided through the `mc_joystick` plugin and offers smooth real-time control of the robot.

- **mc_rtc GUI**: The commanded **X** and **Y** linear velocities, as well as the **yaw** velocity, can also be adjusted manually from the mc_rtc GUI. This is useful for testing policies or operating the robot without a gamepad.

- **Keyboard control**: Keyboard arrow keys can also be used. Currently, only **X** and **Y** velocity control is supported when using the keyboard.

### Configuring Policies

- **Add your policy files** to the [`policy/`](policy/) directory (ONNX format)

- **Configure policy parameters** in [`etc/RLController.in.yaml`](etc/RLController.in.yaml). Each policy can specify:
   - `*robot_name`: Robot name
   - `*use_QP`: QP usage (true/false)
   - `*simulator`: RL env used during training
   - `*used_joints_index`: Joints indices by policy (mc_rtc order)
   - `pd_gains_ratio`: PD gains ratio
   - `*kp, kd`: PD gains (kp and kd)
   - `speed_multiplier_joystick`: max speed of the control (in m/s) when using the joystick plugin
   - `action_scale`: RL action scale (multiplicator)
   - `policy_period_ms`: policy period (ms)

Parameters with "*" are necessary. The others are optional.

- **Define observation vectors** in [`src/utils.cpp`](src/utils.cpp#L131) (l.131). The file includes default examples for:
   - Standing policy for H1 (case 0)
   - Walking policies for H1 (cases 1)

## Advanced Setup

### Adding a New RL Training Environment

Some RL training environments use different joint ordering than the URDF/mc_rtc convention. To add support:

- Define the joint mapping in [`src/PolicySimulatorHandling.h`](src/PolicySimulatorHandling.h) by setting the `mcRtcToSimuIdx_` member variable
- If the mapping is defined in the header, the class will automatically handle unrecognized simulator or robot names
- You can generate automatically the corresponding mapping using the `generate_joint_mapping.py` script, either by modifying the example joints in the script or by specifying source (mc_rtc joint order) and target (RL training env joint order) files. The expected format is one joint name per line.
Example of use :
```bash
./generate_joint_mapping.py --source <mc_rtc_joints_order_file> --target <rl_env_joints_order_file>
```

### Adding a New Robot

To use the controller with a different robot, modify the following:

- **Configuration file** ([`etc/RLController.in.yaml`](etc/RLController.in.yaml#L60) (l.60)) : Add your robot under the `Robot` category with the `mc_rtc_joints_order` corresponding to the joints in URDF order
- **Joint mapping** ([`src/PolicySimulatorHandling.h`](src/PolicySimulatorHandling.h)): Specify the `mcRtcToSimuIdx_` mapping for your robot, similar to adding a new simulator (see [Adding a New RL Training Environment](#adding-a-new-rl-training-environment)).
- **Robot base name** ([`src/RLController.cpp`](src/RLController.cpp#L651)): If the added robot's base is not named 'root', you will need to add a condition here to handle it.