# RLController - RL Policy FSM Controller for H1 Robot

**Important**: This repository is under development, and thus not yet practical to use outside the current example

This FSM controller integrates reinforcement learning policies with mc_rtc for controlling the H1 humanoid robot. It currently supports both a walking and a standing policy.

**Note**: ONNX Runtime is **included** in this repository, so no external installation is required

## Building

```bash
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo

make -j$(nproc)

make install
```

## Usage

### 1. Specify your ONNX model

Default policies are provided in the `policy/` directory.

To specify the used policy, specify its path in the `policy_path` entry of the `etc/RLController.in.yaml` config file.

```yaml
RLController:
  policy_path: "path/to/your/policy.onnx"
  use_async_inference: true  # Optional: enable async inference -> frequence configurable in controller
```

The observation vector is **for now** harcoded and needs to be changed in `utils.cpp`.

### 2. Model Requirements

Currently, only ONNX models are supported, with harcoded observations.
The accepted shapes are :
- **Input tensor**: Observation vector (supports both `[obs_size]` and `[batch, obs_size]` formats)
- **Output tensor**: Action vector (supports both `[action_size]` and `[batch, action_size]` formats)

The controller automatically detects the input/output dimensions from your model.

## Architecture

The controller consists of:

- **RLPolicyInterface**: ONNX model loading and inference
- **RLController**: Main FSM controller integrating RL with mc_rtc
- **State Machines**: Various control states (Standing, Walking, etc.)