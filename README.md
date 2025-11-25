# RLController

An FSM controller that integrates reinforcement learning policies with [mc_rtc](https://jrl-umi3218.github.io/mc_rtc/) for robotic control. This package provides example policies for the H1 humanoid robot and supports ONNX format for policy deployment.

**Note**: ONNX Runtime is bundled with this repository—no external installation required.

## Architecture

The controller is organized into the following components:

- **[etc/RLController.in.yaml](etc/RLController.in.yaml)**: Main configuration file for the controller
- **[RLController](src/RLController.cpp)**: Core FSM controller that integrates RL policies with mc_rtc
- **[RLPolicyInterface](src/RLPolicyInterface.cpp)**: Handles ONNX model loading and inference
- **[PolicySimulatorHandling](src/PolicySimulatorHandling.cpp)**: Manages simulator-specific variations (e.g., joint ordering differences)
- **[states/RL_State](src/states/RL_State.cpp)**: FSM state that executes the RL policy and applies torque commands

## Building

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
make install
```

## Usage

### Robot and Simulator Support

The controller is optimized for the H1 humanoid robot with minimal configuration required. Support for other robots is possible with additional adaptation (see [Adding a New Robot](#adding-a-new-robot)).

Policies trained in ManiSkill and IsaacLab are fully supported. For policies from other training environments, you can add custom simulator support (see [Adding a New Simulator](#adding-a-new-simulator)).

### Policy Management

Default policies are located in the [`policy/`](policy/) directory. The controller supports switching between multiple policies at runtime through the GUI (`RLController/Policy` section).

**Important**: Policy transitions should be compatible with the current state. For example, switching from standing to walking works because the walking policy can handle observations from a standing state, but the reverse may not be true without proper handling.

### Configuring Policies

- **Add your policy files** to the [`policy/`](policy/) directory (ONNX format)

- **Configure policy parameters** in [`etc/RLController.in.yaml`](etc/RLController.in.yaml). Each policy can specify:
   - Robot name
   - Control mode (position/torque)
   - QP usage
   - Simulator used during training
   - Controlled joints indices
   - PD gains ratio
   - PD gains (kp and kd)

- **Define observation vectors** in [`src/utils.cpp`](src/utils.cpp#L131) (l.131). The file includes default examples for:
   - Standing policy (case 0)
   - Walking policies (cases 1-2)

## Advanced Setup

### Adding a New Simulator

Some simulators use different joint ordering than the URDF/mc_rtc convention. To add support:

- Define the joint mapping in [`src/PolicySimulatorHandling.h`](src/PolicySimulatorHandling.h) by setting the `mcRtcToSimuIdx_` member variable
- If the mapping is defined in the header, the class will automatically handle unrecognized simulator or robot names

### Adding a New Robot

To use the controller with a different robot, modify the following:

- **Configuration file** ([`etc/RLController.in.yaml`](etc/RLController.in.yaml#L60) (l.60)) : Add your robot under the `Robot` category with the `mc_rtc_joints_order` corresponding to the joints in URDF order
- **Joint mapping** ([`src/PolicySimulatorHandling.h`](src/PolicySimulatorHandling.h)): Specify the `mcRtcToSimuIdx_` mapping for your robot, similar to adding a new simulator 