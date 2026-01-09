import pybullet as p
import pybullet_data
import numpy as np
import time

class FrankaImpedanceController:
    def __init__(self):
        # Initialize PyBullet
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 1kHz control loop
        self.dt = 1.0/1000.0
        p.setTimeStep(self.dt)

        # Load Robot
        self.panda = p.loadURDF("franka_panda/panda.urdf", [0,0,0], useFixedBase=True)
        self.ee_link = 11 
        
        # Identify all movable joints (usually 9: 7 for arm, 2 for gripper)
        self.movable_joints = []
        for i in range(p.getNumJoints(self.panda)):
            info = p.getJointInfo(self.panda, i)
            if info[2] != p.JOINT_FIXED:
                self.movable_joints.append(i)
        
        self.num_dof = len(self.movable_joints) # Total DoF for the URDF (9)
        self.arm_dof = 7                        # DoF for the Arm only

    
        self.stiffness_diag = np.array([150.0, 150.0, 150.0, 10.0, 10.0, 10.0])
        self.damping_diag = 2.0 * np.sqrt(self.stiffness_diag)
        
        self.K = np.diag(self.stiffness_diag)
        self.D = np.diag(self.damping_diag)

        # Initial Setup: Neutral Pose
        self.q_neutral = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        for i in range(self.arm_dof):
            p.resetJointState(self.panda, self.movable_joints[i], self.q_neutral[i])
            
        # Disable default velocity motors to enable torque control
        p.setJointMotorControlArray(self.panda, self.movable_joints, p.VELOCITY_CONTROL, forces=[0]*self.num_dof)

        # Desired State (Equilibrium is the starting position)
        init_pos, init_orn = self.get_ee_state()
        self.pos_d = np.array(init_pos)
        # Store desired orientation as a rotation matrix
        self.orn_d_mat = np.array(p.getMatrixFromQuaternion(init_orn)).reshape(3,3)
        
        # Safety: Torque rate limiting (1000 Nm/s is Franka's hardware limit)
        self.tau_last = np.zeros(self.arm_dof)
        self.max_delta_tau = 1000.0 * self.dt # Max change allowed per 1ms step

    def get_ee_state(self):
        state = p.getLinkState(self.panda, self.ee_link)
        return np.array(state[0]), np.array(state[1])

    def get_dynamics(self, q_all, dq_all):
        """ 
        Computes Gravity and Coriolis terms separately. 
        PyBullet requires full DoF lists for these calculations.
        """
        # Gravity compensation: Inverse dynamics with zero velocity/acceleration
        tau_grav = np.array(p.calculateInverseDynamics(self.panda, list(q_all), [0.0]*self.num_dof, [0.0]*self.num_dof))
        # Total dynamics: Grav + Coriolis/Centrifugal
        tau_total = np.array(p.calculateInverseDynamics(self.panda, list(q_all), list(dq_all), [0.0]*self.num_dof))
        
        # Return only the arm components (first 7)
        return tau_grav[:self.arm_dof], (tau_total - tau_grav)[:self.arm_dof]

    def compute_orientation_error(self, cur_orn_quat):
        """ 
        1. Get relative rotation R_curr.T * R_d
        2. Extract imaginary part of the resulting quaternion
        3. Rotate error back to the base frame
        """
        R_curr = np.array(p.getMatrixFromQuaternion(cur_orn_quat)).reshape(3,3)
        relative_rot_mat = R_curr.T @ self.orn_d_mat
        
        # Convert matrix to quaternion-style error (axis * sin(theta))
        trace = np.trace(relative_rot_mat)
        if trace > 2.9999: # Small angle approximation
            quat_error = 0.5 * np.array([
                relative_rot_mat[2,1] - relative_rot_mat[1,2],
                relative_rot_mat[0,2] - relative_rot_mat[2,0],
                relative_rot_mat[1,0] - relative_rot_mat[0,1]
            ])
        else:
            # Full extraction for larger errors
            s = 0.5 / np.sqrt(trace + 1.0)
            quat_error = np.array([
                (relative_rot_mat[2,1] - relative_rot_mat[1,2]) * s,
                (relative_rot_mat[0,2] - relative_rot_mat[2,0]) * s,
                (relative_rot_mat[1,0] - relative_rot_mat[0,1]) * s
            ])
        
        # Map error from local EE frame back to Base frame
        return -R_curr @ quat_error

    def step(self):
        # 1. Sense: Get full state for all joints (including gripper)
        joint_states = p.getJointStates(self.panda, self.movable_joints)
        q_all = np.array([s[0] for s in joint_states])
        dq_all = np.array([s[1] for s in joint_states])
        
        # Subsets for the 7-DOF arm
        q = q_all[:self.arm_dof]
        dq = dq_all[:self.arm_dof]
        
        pos, orn = self.get_ee_state()
        
        # 2. Kinematics: Calculate Jacobian for all movable joints
        res = p.calculateJacobian(self.panda, self.ee_link, [0, 0, 0], list(q_all), list(dq_all), [0.0]*self.num_dof)
        J_full = np.vstack(res)
        J = J_full[:, :self.arm_dof] # Slice for the 7 arm joints
        
        # 3. Compute Cartesian Errors
        error = np.zeros(6)
        error[:3] = pos - self.pos_d
        error[3:] = self.compute_orientation_error(orn)
        
        # 4. Control Law 
        # tau_task = J^T * (-K*error - D*(J*dq))
        dx = J @ dq
        tau_task = J.T @ (-self.K @ error - self.D @ dx)
        
        # 5. Dynamics: Add Gravity (for Sim) and Coriolis (Official)
        tau_grav, tau_coriolis = self.get_dynamics(q_all, dq_all)
        tau_d = tau_task + tau_coriolis + tau_grav
        
        # 6. Apply Torque Rate Limiting (Safety/Hardware Fidelity)
        diff = np.clip(tau_d - self.tau_last, -self.max_delta_tau, self.max_delta_tau)
        tau_saturated = self.tau_last + diff
        self.tau_last = tau_saturated

        # 7. Execute: Apply torques only to the arm
        p.setJointMotorControlArray(
            self.panda, 
            self.movable_joints[:self.arm_dof], 
            p.TORQUE_CONTROL, 
            forces=tau_saturated
        )
        
        p.stepSimulation()

if __name__ == "__main__":
    controller = FrankaImpedanceController()

    print("Use Ctrl+Mouse to pull on the robot and test compliance.")

    try:
        while True:
            controller.step()
            time.sleep(controller.dt)
    except KeyboardInterrupt:
        p.disconnect()