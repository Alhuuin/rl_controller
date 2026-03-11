#!/usr/bin/env python3
import argparse
import sys


DEFAULT_MC_RTC = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

DEFAULT_RL = [
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "torso_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_joint",
    "right_ankle_joint",
    "left_elbow_joint",
    "right_elbow_joint",
]


def load_joint_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def compute_joint_mapping(source, target):
    source_index = {name: i for i, name in enumerate(source)}
    mapping = []

    for name in target:
        if name not in source_index:
            raise ValueError(f"Joint '{name}' not found in source list")
        mapping.append(source_index[name])

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Generate a std::vector<int> joint mapping from two joint-name lists."
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Path to file containing source joint list (one joint name per line)",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Path to file containing target joint list (one joint name per line)",
    )

    args = parser.parse_args()

    if args.source:
        source = load_joint_list(args.source)
        print(f"[INFO] Using source joint list from file: {args.source}")
    else:
        source = DEFAULT_MC_RTC
        print("[INFO] Using DEFAULT source joint list")

    if args.target:
        target = load_joint_list(args.target)
        print(f"[INFO] Using target joint list from file: {args.target}")
    else:
        target = DEFAULT_RL
        print("[INFO] Using DEFAULT target joint list")

    mapping = compute_joint_mapping(source, target)

    print("\nstd::vector<int> joint_mapping = {", end="")
    print(", ".join(map(str, mapping)), end="")
    print("};")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
