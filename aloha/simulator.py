"""MuJoCo robot simulator with position control for ALOHA dual-arm manipulator.

This simulator is designed for the ALOHA robot which has:
- Two fixed arms (left and right) mounted on a table
- Each arm has 6 DOF (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
- Each arm has a 2-finger parallel gripper

Currently refactored with ArmController.
"""

import numpy as np
import mujoco, mujoco.viewer
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple

from config import RobotConfig
from arm_controller import ArmController


class MujocoSimulator:
    """MuJoCo simulator with position control for ALOHA robot."""

    def __init__(self, xml_path: str = "../model/aloha/scene.xml") -> None:
        """Initialize simulator with MuJoCo model and control indices."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Initialize Left Arm
        self.left_arm = ArmController(
            model=self.model,
            data=self.data,
            joint_names=RobotConfig.LEFT_ARM_JOINT_NAMES,
            actuator_names=RobotConfig.LEFT_ARM_ACTUATOR_NAMES,
            ee_site_name=RobotConfig.LEFT_EE_SITE_NAME,
            gripper_joint_name=RobotConfig.LEFT_GRIPPER_JOINT_NAME,
            gripper_actuator_name=RobotConfig.LEFT_GRIPPER_ACTUATOR_NAME,
            init_position=RobotConfig.ARM_INIT_POSITION,
            init_gripper_width=RobotConfig.GRIPPER_INIT_WIDTH,
            default_ee_target_ori=RobotConfig.LEFT_ARM_DEFAULT_TARGET_ORI
        )

        # Initialize Right Arm (Not actively commanded by default backward compatible APIs, but forced to initial states)
        self.right_arm = ArmController(
            model=self.model,
            data=self.data,
            joint_names=RobotConfig.RIGHT_ARM_JOINT_NAMES,
            actuator_names=RobotConfig.RIGHT_ARM_ACTUATOR_NAMES,
            ee_site_name=RobotConfig.RIGHT_EE_SITE_NAME,
            gripper_joint_name=RobotConfig.RIGHT_GRIPPER_JOINT_NAME,
            gripper_actuator_name=RobotConfig.RIGHT_GRIPPER_ACTUATOR_NAME,
            init_position=RobotConfig.ARM_INIT_POSITION,
            init_gripper_width=RobotConfig.GRIPPER_INIT_WIDTH,
            default_ee_target_ori=RobotConfig.RIGHT_ARM_DEFAULT_TARGET_ORI
        )

        # Compute initial forward kinematics
        mujoco.mj_forward(self.model, self.data)

    # ============================================================
    # Backward Compatibility Methods (Delegates to left or right arm)
    # ============================================================

    def _get_arm(self, arm: str):
        if arm == 'left':
            return self.left_arm
        elif arm == 'right':
            return self.right_arm
        else:
            raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")

    def get_arm_target_joint(self, arm: str = 'left') -> np.ndarray:
        return self._get_arm(arm).get_arm_target_joint()

    def set_arm_target_joint(self, arm_target_joint: np.ndarray, arm: str = 'left') -> None:
        self._get_arm(arm).set_arm_target_joint(arm_target_joint)

    def get_arm_joint_position(self, arm: str = 'left') -> np.ndarray:
        return self._get_arm(arm).get_arm_joint_position()

    def get_arm_joint_diff(self, arm: str = 'left') -> np.ndarray:
        return self._get_arm(arm).get_arm_joint_diff()

    def get_arm_joint_velocity(self, arm: str = 'left') -> np.ndarray:
        return self._get_arm(arm).get_arm_joint_velocity()

    def get_ee_position(self, data: Optional[mujoco.MjData] = None, arm: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
        return self._get_arm(arm).get_ee_position(data)

    def set_ee_target_position(self, target_pos: np.ndarray, target_ori: Optional[np.ndarray] = None, arm: str = 'left') -> Tuple[bool, np.ndarray]:
        return self._get_arm(arm).set_ee_target_position(target_pos, target_ori)

    def get_gripper_width(self, arm: str = 'left') -> float:
        return self._get_arm(arm).get_gripper_width()

    def set_target_gripper_width(self, width: float, arm: str = 'left') -> None:
        self._get_arm(arm).set_target_gripper_width(width)

    def get_gripper_width_diff(self, arm: str = 'left') -> float:
        return self._get_arm(arm).get_gripper_width_diff()

    def pick_object(self, *args, arm: str = 'left', **kwargs) -> bool:
        return self._get_arm(arm).pick_object(*args, **kwargs)

    def place_object(self, *args, arm: str = 'left', **kwargs) -> bool:
        return self._get_arm(arm).place_object(*args, **kwargs)

    # ============================================================
    # Object Interaction Methods
    # ============================================================

    @staticmethod
    def _rotation_matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
        return R.from_matrix(rot.reshape(3, 3)).as_euler("xyz")

    def get_object_positions(self) -> dict:
        objects = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("object_"):
                objects[name] = {
                    'id': i,
                    'pos': self.data.xpos[i].copy(),
                    'ori': self._rotation_matrix_to_euler_xyz(self.data.xmat[i])
                }
        return objects

    # ============================================================
    # Simulation Loop
    # ============================================================

    def run(self) -> None:
        """Run simulation with 3D viewer and control loop (blocking)."""
        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            # Camera setup
            v.cam.lookat[:] = RobotConfig.CAM_LOOKAT
            v.cam.distance = RobotConfig.CAM_DISTANCE
            v.cam.azimuth = RobotConfig.CAM_AZIMUTH
            v.cam.elevation = RobotConfig.CAM_ELEVATION

            # Hide debug visuals
            v.opt.geomgroup[0] = 0
            v.opt.sitegroup[0] = v.opt.sitegroup[1] = v.opt.sitegroup[2] = 0
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False
            v.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            v.opt.label = mujoco.mjtLabel.mjLABEL_NONE

            # Main loop
            while v.is_running():
                # Apply arm controllers
                self.left_arm.update_control_loop()
                self.right_arm.update_control_loop()

                mujoco.mj_step(self.model, self.data)
                v.sync()
