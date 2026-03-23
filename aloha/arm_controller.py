import time
import numpy as np
import mujoco
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R

from config import RobotConfig


# ============================================================
# Dual Quaternion 유틸리티
# ============================================================

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _dq_mul(dq1, dq2):
    r1, d1 = dq1[:4], dq1[4:]
    r2, d2 = dq2[:4], dq2[4:]
    return np.concatenate([
        _quat_mul(r1, r2),
        _quat_mul(r1, d2) + _quat_mul(d1, r2),
    ])

def _dq_conj(dq):
    r, d = dq[:4], dq[4:]
    return np.concatenate([
        [r[0], -r[1], -r[2], -r[3]],
        [d[0], -d[1], -d[2], -d[3]],
    ])

def _mat_pos_to_dq(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """3x3 회전행렬 + 위치벡터 → 듀얼 쿼터니언"""
    from scipy.spatial.transform import Rotation as R
    q_r = R.from_matrix(rot).as_quat()          # [x, y, z, w]
    q_r = np.array([q_r[3], q_r[0], q_r[1], q_r[2]])  # → [w, x, y, z]
    q_d = 0.5 * _quat_mul(np.array([0, *pos]), q_r)
    return np.concatenate([q_r, q_d])

def _dq_error_to_6d(dq_err: np.ndarray) -> np.ndarray:
    """dq_error → [pos_error(3), ori_error(3)]"""
    ori_error = 2.0 * dq_err[1:4]
    # 정확한 위치 추출: t = 2 * q_d * conj(q_r)
    q_r_conj = np.array([dq_err[0], -dq_err[1], -dq_err[2], -dq_err[3]])
    pos_error = 2.0 * _quat_mul(dq_err[4:], q_r_conj)[1:]
    return np.concatenate([pos_error, ori_error])


class ArmController:
    """Controller for a single ALOHA arm and gripper."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: list,
        actuator_names: list,
        ee_site_name: str,
        gripper_joint_name: str,
        gripper_actuator_name: str,
        init_position: np.ndarray,
        init_gripper_width: float,
        default_ee_target_ori: np.ndarray
    ) -> None:
        self.model = model
        self.data = data
        self.dt = self.model.opt.timestep

        self._arm_target_joint = np.array(init_position).copy()
        self._current_target_joint = np.array(init_position).copy()
        self._gripper_target_width = init_gripper_width
        self._arm_error_integral = np.zeros(6,)
        self.default_ee_target_ori = default_ee_target_ori

        # Resolve arm joint IDs and actuator IDs
        self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        self.arm_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]
        
        # Build DOF indices for arm joints
        self.arm_dof_indices = []
        for joint_id in self.arm_joint_ids:
            dof_adr = self.model.jnt_dofadr[joint_id]
            dof_num = self._get_joint_dof_count(joint_id)
            self.arm_dof_indices.extend(range(dof_adr, dof_adr + dof_num))

        # Resolve end effector and gripper IDs
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        self.gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator_name)
        self.gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, gripper_joint_name)

        # Basic Initialization (Sets positions directly)
        self.reset_to_init(init_position, init_gripper_width)

    def _get_joint_dof_count(self, joint_id: int) -> int:
        """Get the number of DOFs for a joint."""
        joint_type = self.model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE: return 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL: return 3
        if joint_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE): return 1
        raise ValueError(f"Unsupported joint type for joint_id {joint_id}")

    def reset_to_init(self, init_position: np.ndarray, init_gripper_width: float) -> None:
        """Force the joint states to the initial configuration."""
        for i, (joint_id, actuator_id) in enumerate(zip(self.arm_joint_ids, self.arm_actuator_ids)):
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = init_position[i]
            self.data.ctrl[actuator_id] = init_position[i]

        gripper_qpos_adr = self.model.jnt_qposadr[self.gripper_joint_id]
        self.data.qpos[gripper_qpos_adr] = init_gripper_width
        self.data.ctrl[self.gripper_actuator_id] = init_gripper_width

    # ============================================================
    # Arm Joint Control Methods
    # ============================================================

    def get_arm_target_joint(self) -> np.ndarray:
        return self._arm_target_joint

    def set_arm_target_joint(self, arm_target_joint: np.ndarray) -> None:
        self._arm_target_joint = np.array(arm_target_joint)
        self._arm_error_integral[:] = 0

    def get_arm_joint_position(self) -> np.ndarray:
        positions = []
        for joint_id in self.arm_joint_ids:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            positions.append(self.data.qpos[qpos_adr])
        return np.array(positions)

    def get_arm_joint_diff(self) -> np.ndarray:
        return self._arm_target_joint - self.get_arm_joint_position()

    def get_arm_joint_velocity(self) -> np.ndarray:
        velocities = []
        for joint_id in self.arm_joint_ids:
            dof_adr = self.model.jnt_dofadr[joint_id]
            velocities.append(self.data.qvel[dof_adr])
        return np.array(velocities)

    def _compute_arm_control(self) -> np.ndarray:
        # Smooth update
        diff = self._arm_target_joint - self._current_target_joint
        dist = np.linalg.norm(diff)
        
        if dist > 1e-5:
            max_step = RobotConfig.MAX_ARM_SPEED * self.dt
            if dist < max_step:
                self._current_target_joint = self._arm_target_joint.copy()
            else:
                self._current_target_joint += (diff / dist) * max_step
        
        current_pos = self.get_arm_joint_position()
        current_vel = self.get_arm_joint_velocity()
        pos_error = self._current_target_joint - current_pos

        self._arm_error_integral += pos_error * self.dt
        self._arm_error_integral = np.clip(
            self._arm_error_integral,
            -RobotConfig.ARM_I_LIMIT,
            RobotConfig.ARM_I_LIMIT
        )

        p_term = RobotConfig.ARM_KP * pos_error
        i_term = RobotConfig.ARM_KI * self._arm_error_integral
        d_term = RobotConfig.ARM_KD * current_vel

        return current_pos + p_term + i_term - d_term

    # ============================================================
    # End Effector Control Methods
    # ============================================================

    @staticmethod
    def _rotation_matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R
        return R.from_matrix(rot.reshape(3, 3)).as_euler("xyz")

    def get_ee_position(self, data: Optional[mujoco.MjData] = None) -> Tuple[np.ndarray, np.ndarray]:
        if data is None:
            data = self.data
        ee_pos = data.site_xpos[self.ee_site_id].copy()
        ee_rot = data.site_xmat[self.ee_site_id]
        ee_ori = self._rotation_matrix_to_euler_xyz(ee_rot)
        return ee_pos, ee_ori

    def _compute_ee_jacobian(self, data: Optional[mujoco.MjData] = None) -> np.ndarray:
        if data is None:
            data = self.data
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_site_id)
        jacp_arm = jacp[:, self.arm_dof_indices]
        jacr_arm = jacr[:, self.arm_dof_indices]
        return np.vstack([jacp_arm, jacr_arm])

    def _compute_dq_jacobian(self, data: Optional[mujoco.MjData] = None) -> np.ndarray:
        """
        Screw (DQ) Jacobian: 6×n, 각 컬럼 = [p_i × n_i; n_i]
        출력: [pos(3); ori(3)] 순서 → _dq_error_to_6d 반환 순서와 일치
        """
        if data is None:
            data = self.data
        n = len(self.arm_joint_ids)
        J = np.zeros((6, n))
        ee_pos = data.site_xpos[self.ee_site_id]
        for i, joint_id in enumerate(self.arm_joint_ids):
            dof_adr = self.model.jnt_dofadr[joint_id]
            n_i = data.xaxis[dof_adr]         # 월드 프레임 관절 축
            p_i = data.xanchor[dof_adr]       # 월드 프레임 관절 위치
            J[:3, i] = np.cross(n_i, ee_pos - p_i)  # 이동 기여: n × (p_ee - p_i)
            J[3:, i] = n_i                    # 회전 기여 (ori)
        return J

    def _solve_ik_position(
        self, 
        target_pos: np.ndarray, 
        target_ori: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None
    ) -> Tuple[bool, np.ndarray]:
        if max_iterations is None:
            max_iterations = RobotConfig.IK_MAX_ITERATIONS

        q = self.get_arm_joint_position().copy()

        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos[:]
        mujoco.mj_forward(self.model, ik_data)

        if target_ori is None:
            target_mat = ik_data.site_xmat[self.ee_site_id].copy().reshape(3, 3)
        else:
            target_mat = R.from_euler("xyz", target_ori).as_matrix()

        # --- Dual Quaternion 기반 오차 계산 ---
        dq_target = _mat_pos_to_dq(target_mat, target_pos)

        for _ in range(max_iterations):
            for i, joint_id in enumerate(self.arm_joint_ids):
                qpos_adr = self.model.jnt_qposadr[joint_id]
                ik_data.qpos[qpos_adr] = q[i]
            mujoco.mj_forward(self.model, ik_data)

            current_pos = ik_data.site_xpos[self.ee_site_id].copy()
            current_mat = ik_data.site_xmat[self.ee_site_id].copy().reshape(3, 3)

            pos_error = target_pos - current_pos
            err_rot = R.from_matrix(target_mat @ current_mat.T)
            ori_error = err_rot.as_rotvec()
            error = np.concatenate([pos_error, ori_error])

            if (np.linalg.norm(pos_error) < RobotConfig.IK_POSITION_TOLERANCE and
                np.linalg.norm(ori_error) < RobotConfig.IK_ORIENTATION_TOLERANCE):
                return True, q

            jacobian = self._compute_ee_jacobian(ik_data)
            jjt = jacobian @ jacobian.T
            damping = (RobotConfig.IK_DAMPING ** 2) * np.eye(jacobian.shape[0])
            inv_term = np.linalg.inv(jjt + damping)
            dq = jacobian.T @ (inv_term @ error)
            q += RobotConfig.IK_STEP_SIZE * dq
            q = np.clip(q, RobotConfig.ARM_JOINT_LIMITS[:, 0], RobotConfig.ARM_JOINT_LIMITS[:, 1])

        return False, q

    def set_ee_target_position(self, target_pos: np.ndarray, target_ori: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray]:
        success, joint_angles = self._solve_ik_position(target_pos, target_ori=target_ori)
        if success:
            self.set_arm_target_joint(joint_angles)
        return success, joint_angles

    # ============================================================
    # Gripper Control Methods
    # ============================================================

    def get_gripper_width(self) -> float:
        gripper_qpos_adr = self.model.jnt_qposadr[self.gripper_joint_id]
        return 2.0 * self.data.qpos[gripper_qpos_adr]

    def set_target_gripper_width(self, width: float) -> None:
        self._gripper_target_width = np.clip(width / 2.0, 0.002, 0.037)

    def get_gripper_width_diff(self) -> float:
        return self._gripper_target_width * 2.0 - self.get_gripper_width()

    def _compute_gripper_control(self) -> float:
        return self._gripper_target_width

    # ============================================================
    # Pick, Place & Control Step
    # ============================================================

    def _wait_for_arm_convergence(self, timeout: float = 10.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            pos_error = np.linalg.norm(self.get_arm_joint_diff())
            vel_error = np.linalg.norm(self.get_arm_joint_velocity())
            if pos_error < 0.1 and vel_error < 0.1:
                return True
            time.sleep(0.02)
        return False

    def pick_object(self, object_pos: np.ndarray, approach_height: float = 0.1, lift_height: float = 0.2, 
                    return_to_home: bool = True, timeout: float = 10.0, verbose: bool = False) -> bool:
        if verbose: print(f"Starting pick sequence at position [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]")

        target_ori = self.default_ee_target_ori

        if verbose: print("  Step 1: Opening gripper...")
        self.set_target_gripper_width(0.074)
        time.sleep(1.0)

        approach_pos = np.array([object_pos[0], object_pos[1], object_pos[2] + approach_height])
        if verbose: print(f"  Step 2: Moving to approach position (height: {approach_height:.3f}m above object)...")
        success, _ = self.set_ee_target_position(approach_pos, target_ori=target_ori)
        if not success:
            if verbose: print("  Failed to reach approach position")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose: print("  Timeout waiting for approach position")
            return False

        grasp_pos = np.array([object_pos[0], object_pos[1], object_pos[2] + 0.01])
        if verbose: print(f"  Step 3: Lowering to grasp position...")
        success, _ = self.set_ee_target_position(grasp_pos, target_ori=target_ori)
        if not success:
            if verbose: print("  Failed to reach grasp position")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose: print("  Timeout waiting for grasp position")
            return False

        if verbose: print("  Step 4: Closing gripper to grasp...")
        self.set_target_gripper_width(0.01)
        time.sleep(1.5)

        lift_pos = np.array([object_pos[0], object_pos[1], object_pos[2] + lift_height])
        if verbose: print(f"  Step 5: Lifting object (height: {lift_height:.3f}m above original position)...")
        success, _ = self.set_ee_target_position(lift_pos, target_ori=target_ori)
        if not success:
            if verbose: print("  Failed to lift object")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose: print("  Timeout waiting for lift position")
            return False

        if return_to_home:
            if verbose: print("  Step 6: Returning arm to home position...")
            self.set_arm_target_joint(RobotConfig.ARM_INIT_POSITION)
            if not self._wait_for_arm_convergence(timeout):
                if verbose: print("  Timeout waiting for home position")
                return False

        if verbose: print("  Pick sequence completed successfully!")
        return True

    def place_object(self, place_pos: np.ndarray, approach_height: float = 0.2, retract_height: float = 0.3, 
                     return_to_home: bool = True, timeout: float = 10.0, verbose: bool = False) -> bool:
        if verbose: print(f"Starting place sequence at position [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")

        target_ori = self.default_ee_target_ori

        approach_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + approach_height])
        if verbose: print(f"  Step 1: Moving to approach position (height: {approach_height:.3f}m above target)...")
        success, _ = self.set_ee_target_position(approach_pos, target_ori=target_ori)
        if not success:
            if verbose: print("  Failed to reach approach position")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose: print("  Timeout waiting for approach position")
            return False

        if verbose: print("  Step 2: Opening gripper to release object...")
        self.set_target_gripper_width(0.074)
        time.sleep(1.5)

        retract_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + retract_height])
        if verbose: print(f"  Step 3: Retracting (height: {retract_height:.3f}m above placement)...")
        success, _ = self.set_ee_target_position(retract_pos, target_ori=target_ori)
        if not success:
            if verbose: print("  Failed to retract")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose: print("  Timeout waiting for retract position")
            return False

        if return_to_home:
            if verbose: print("  Step 4: Returning arm to home position...")
            self.set_arm_target_joint(RobotConfig.ARM_INIT_POSITION)
            if not self._wait_for_arm_convergence(timeout):
                if verbose: print("  Timeout waiting for home position")
                return False

        if verbose: print("  Place sequence completed successfully!")
        return True

    def bimanual_move_object(
        self,
        other_arm: 'ArmController',
        target_pos: np.ndarray,
        target_ori: np.ndarray,
        dq_rel: np.ndarray,
        timeout: float = 10.0,
        verbose: bool = False
    ) -> Tuple[bool, bool]:
        """두 팔이 물체를 잡은 상태에서 목표 포즈로 이동.

        Args:
            other_arm: follower 팔
            target_pos: leader(self) EE 목표 위치
            target_ori: leader(self) EE 목표 방향 (euler xyz)
            dq_rel: 두 EE 간 고정 상대 포즈 (잡는 순간 측정값)
            timeout: 수렴 대기 시간
            verbose: 진행 상황 출력

        Returns:
            (leader 성공, follower 성공)
        """
        from scipy.spatial.transform import Rotation as R

        # leader IK
        if verbose: print("  [bimanual] Leader arm solving IK...")
        success_leader, _ = self.set_ee_target_position(target_pos, target_ori=target_ori)
        if not success_leader:
            if verbose: print("  [bimanual] Leader IK failed")
            return False, False

        # follower 목표 계산: dq_follower = dq_leader * dq_rel
        target_mat = R.from_euler("xyz", target_ori).as_matrix()
        dq_leader = _mat_pos_to_dq(target_mat, target_pos)
        dq_follower = _dq_mul(dq_leader, dq_rel)

        # follower DQ → pos, ori 추출
        q_r = dq_follower[:4]
        q_r_conj = np.array([q_r[0], -q_r[1], -q_r[2], -q_r[3]])
        follower_pos = 2.0 * _quat_mul(dq_follower[4:], q_r_conj)[1:]
        follower_ori = R.from_quat([q_r[1], q_r[2], q_r[3], q_r[0]]).as_euler("xyz")

        # follower IK
        if verbose: print("  [bimanual] Follower arm solving IK...")
        success_follower, _ = other_arm.set_ee_target_position(follower_pos, target_ori=follower_ori)
        if not success_follower:
            if verbose: print("  [bimanual] Follower IK failed")
            return success_leader, False

        # 두 팔 수렴 대기
        conv_leader   = self._wait_for_arm_convergence(timeout)
        conv_follower = other_arm._wait_for_arm_convergence(timeout)

        if verbose:
            print(f"  [bimanual] Leader: {'OK' if conv_leader else 'TIMEOUT'}, "
                  f"Follower: {'OK' if conv_follower else 'TIMEOUT'}")

        return conv_leader, conv_follower

    def update_control_loop(self) -> None:
        """Apply computed targets to the simulator data (called every step)."""
        arm_control = self._compute_arm_control()
        for i, actuator_id in enumerate(self.arm_actuator_ids):
            self.data.ctrl[actuator_id] = arm_control[i]

        gripper_control = self._compute_gripper_control()
        self.data.ctrl[self.gripper_actuator_id] = gripper_control
