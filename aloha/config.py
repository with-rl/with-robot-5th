import numpy as np

class RobotConfig:
    """Robot simulation configuration constants for ALOHA."""

    # Left arm joints: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
    LEFT_ARM_JOINT_NAMES = [
        "left/waist",
        "left/shoulder",
        "left/elbow",
        "left/forearm_roll",
        "left/wrist_angle",
        "left/wrist_rotate"
    ]

    LEFT_ARM_ACTUATOR_NAMES = [
        "left/waist",
        "left/shoulder",
        "left/elbow",
        "left/forearm_roll",
        "left/wrist_angle",
        "left/wrist_rotate"
    ]

    # Right arm joints: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
    RIGHT_ARM_JOINT_NAMES = [
        "right/waist",
        "right/shoulder",
        "right/elbow",
        "right/forearm_roll",
        "right/wrist_angle",
        "right/wrist_rotate"
    ]

    RIGHT_ARM_ACTUATOR_NAMES = [
        "right/waist",
        "right/shoulder",
        "right/elbow",
        "right/forearm_roll",
        "right/wrist_angle",
        "right/wrist_rotate"
    ]

    # End effector site name
    LEFT_EE_SITE_NAME = "left/gripper"
    RIGHT_EE_SITE_NAME = "right/gripper"

    # Gripper actuator/joint names
    LEFT_GRIPPER_ACTUATOR_NAME = "left/gripper"
    LEFT_GRIPPER_JOINT_NAME = "left/left_finger"
    RIGHT_GRIPPER_ACTUATOR_NAME = "right/gripper"
    RIGHT_GRIPPER_JOINT_NAME = "right/left_finger"

    # Default Target Orientation for downward grasp
    LEFT_ARM_DEFAULT_TARGET_ORI = np.array([0, np.pi/2, 0])
    RIGHT_ARM_DEFAULT_TARGET_ORI = np.array([0, np.pi/2, np.pi])

    # Arm PID controller gains for position control (6 joints)
    ARM_KP = np.array([2.0, 2.0, 2.0, 1.5, 1.0, 1.0])
    ARM_KI = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
    ARM_I_LIMIT = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    ARM_KD = np.array([0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
    
    # Joint limits from aloha.xml
    ARM_JOINT_LIMITS = np.array([
        [-3.14158, 3.14158],   # waist
        [-1.85005, 1.25664],   # shoulder
        [-1.76278, 1.6057],    # elbow
        [-3.14158, 3.14158],   # forearm_roll
        [-1.8675, 2.23402],    # wrist_angle
        [-3.14158, 3.14158]    # wrist_rotate
    ])

    # IK solver parameters
    IK_MAX_ITERATIONS = 100
    IK_POSITION_TOLERANCE = 0.001  # 1mm
    IK_ORIENTATION_TOLERANCE = 0.01  # ~0.57 degrees
    IK_DAMPING = 0.01  # Damped Least Squares damping factor
    IK_STEP_SIZE = 0.5  # Step size for joint updates

    # Camera settings
    CAM_LOOKAT = [0.0, -0.1, 0.2]
    CAM_DISTANCE = 1.5
    CAM_AZIMUTH = 90
    CAM_ELEVATION = -20

    # Initial positions
    ARM_INIT_POSITION = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    GRIPPER_INIT_WIDTH = 0.037  # Fully open (max is 0.037)

    # Maximum arm joint speed in rad/s (for smoothing)
    MAX_ARM_SPEED = 2.0  # Limits the rate of change of target position
