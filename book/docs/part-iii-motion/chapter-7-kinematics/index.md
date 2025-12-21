---
title: Chapter 7 - Kinematics and Dynamics
sidebar_position: 1
---

# Chapter 7: Kinematics and Dynamics

## Learning Goals

- Master forward and inverse kinematics
- Understand robot dynamics and motion planning
- Learn trajectory generation techniques
- Implement kinematic solvers for robotic arms
- Generate smooth trajectories for robot motion
- Control robot joints with precise positioning

## Introduction to Robot Kinematics

Robot kinematics is the study of motion in robotic systems without considering the forces that cause the motion. It's divided into two main areas:

1. **Forward Kinematics**: Given joint angles, calculate the position and orientation of the end-effector
2. **Inverse Kinematics**: Given a desired end-effector position and orientation, calculate the required joint angles

Kinematics is fundamental to robot control, enabling precise positioning and motion planning.

### Coordinate Systems and Transformations

Robots operate in 3D space using various coordinate systems:

- **World Frame**: Fixed reference frame for the entire workspace
- **Base Frame**: Attached to the robot's base
- **Joint Frames**: Attached to each joint
- **End-Effector Frame**: Attached to the robot's tool or gripper

Homogeneous transformation matrices are used to represent position and orientation in a single 4×4 matrix:

```python
import numpy as np


def create_rotation_matrix(roll, pitch, yaw):
    """Create rotation matrix from roll, pitch, yaw angles"""
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R


def create_homogeneous_transform(translation, rotation_matrix):
    """Create 4x4 homogeneous transformation matrix"""
    T = np.eye(4)
    T[0:3, 0:3] = rotation_matrix
    T[0:3, 3] = translation
    return T


def transform_point(point, transform_matrix):
    """Transform a 3D point using a 4x4 transformation matrix"""
    # Convert to homogeneous coordinates
    homogeneous_point = np.append(point, 1)
    # Apply transformation
    transformed_point = transform_matrix @ homogeneous_point
    # Convert back to 3D
    return transformed_point[:3]


# Example usage
def main():
    # Create a transformation: translate by (1, 2, 3) and rotate by 45 degrees around Z-axis
    translation = np.array([1.0, 2.0, 3.0])
    rotation_matrix = create_rotation_matrix(0, 0, np.pi/4)  # 45 degrees around Z
    T = create_homogeneous_transform(translation, rotation_matrix)

    # Transform a point
    original_point = np.array([1.0, 0.0, 0.0])
    transformed_point = transform_point(original_point, T)

    print(f"Original point: {original_point}")
    print(f"Transformed point: {transformed_point}")


if __name__ == '__main__':
    main()
```

## Forward Kinematics

Forward kinematics calculates the end-effector position and orientation given joint angles. For serial manipulators, this is computed using the Denavit-Hartenberg (DH) parameters.

### Denavit-Hartenberg Parameters

The DH convention provides a systematic way to define coordinate frames on a robotic manipulator:

```python
import numpy as np


class DHParameter:
    def __init__(self, a, alpha, d, theta):
        """
        Denavit-Hartenberg parameters:
        a: link length
        alpha: link twist
        d: link offset
        theta: joint angle
        """
        self.a = a
        self.alpha = alpha
        self.d = d
        self.theta = theta

    def get_transformation_matrix(self):
        """Calculate the transformation matrix for this joint"""
        # Calculate transformation matrix based on DH parameters
        sa = np.sin(self.alpha)
        ca = np.cos(self.alpha)
        st = np.sin(self.theta)
        ct = np.cos(self.theta)

        T = np.array([
            [ct, -st*ca, st*sa, self.a*ct],
            [st, ct*ca, -ct*sa, self.a*st],
            [0, sa, ca, self.d],
            [0, 0, 0, 1]
        ])
        return T


class ForwardKinematics:
    def __init__(self, dh_parameters):
        """
        Initialize with DH parameters for each joint
        dh_parameters: list of DHParameter objects
        """
        self.dh_params = dh_parameters

    def calculate_transform(self, joint_angles):
        """
        Calculate the transformation from base to end-effector
        joint_angles: list of joint angles (for revolute joints)
        """
        if len(joint_angles) != len(self.dh_params):
            raise ValueError(f"Number of joint angles ({len(joint_angles)}) must match number of joints ({len(self.dh_params)})")

        # Update theta values in DH parameters
        dh_params = []
        for i, (param, angle) in enumerate(zip(self.dh_params, joint_angles)):
            new_param = DHParameter(param.a, param.alpha, param.d, angle)
            dh_params.append(new_param)

        # Calculate cumulative transformation
        T_total = np.eye(4)
        for param in dh_params:
            T = param.get_transformation_matrix()
            T_total = T_total @ T

        return T_total

    def get_end_effector_position(self, joint_angles):
        """Get only the position of the end-effector"""
        T = self.calculate_transform(joint_angles)
        return T[0:3, 3]

    def get_end_effector_pose(self, joint_angles):
        """Get both position and orientation of the end-effector"""
        T = self.calculate_transform(joint_angles)
        position = T[0:3, 3]
        orientation = T[0:3, 0:3]
        return position, orientation


# Example: 3-DOF planar manipulator
def main():
    # Define DH parameters for a simple 3-DOF planar manipulator
    dh_params = [
        DHParameter(a=1.0, alpha=0, d=0, theta=0),  # Joint 1
        DHParameter(a=1.0, alpha=0, d=0, theta=0),  # Joint 2
        DHParameter(a=0.5, alpha=0, d=0, theta=0)   # Joint 3 (end-effector)
    ]

    fk = ForwardKinematics(dh_params)

    # Calculate pose for a specific joint configuration
    joint_angles = [np.pi/4, np.pi/6, -np.pi/3]  # 45°, 30°, -60°
    position, orientation = fk.get_end_effector_pose(joint_angles)

    print(f"Joint angles: {np.degrees(joint_angles)} degrees")
    print(f"End-effector position: {position}")
    print(f"End-effector orientation:\n{orientation}")


if __name__ == '__main__':
    main()
```

### Using Modern Robotics Library (Conceptual)

For more complex robots, libraries like `modern_robotics` provide efficient implementations:

```python
# This is a conceptual example - in practice, you'd use the modern_robotics library
import numpy as np


class ModernRoboticsFK:
    def __init__(self, screw_axes, joint_limits):
        """
        Initialize with screw axes and joint limits
        screw_axes: 6xN matrix where each column is a screw axis
        joint_limits: list of (min, max) joint limits
        """
        self.screw_axes = np.array(screw_axes)  # Each column is a twist (v, omega)
        self.joint_limits = joint_limits
        self.N = self.screw_axes.shape[1]  # Number of joints

    def matrix_exp6(self, se3_matrix):
        """Matrix exponential for se(3)"""
        # Extract angular and linear parts
        omega = se3_matrix[0:3, 0:3]
        v = se3_matrix[0:3, 3]

        # Calculate rotation matrix
        theta = np.linalg.norm(omega[2, 1], omega[0, 2], omega[1, 0])
        if theta < 1e-6:
            # Small angle approximation
            R = np.eye(3) + omega
            p = v
        else:
            # General case
            omega_skew = np.array([[0, -omega[2, 1], omega[1, 2]],
                                  [omega[2, 1], 0, -omega[0, 2]],
                                  [-omega[1, 2], omega[0, 2], 0]])
            omega_skew_sq = omega_skew @ omega_skew

            R = np.eye(3) + np.sin(theta) * omega_skew + (1 - np.cos(theta)) * omega_skew_sq
            G_inv = (np.eye(3) - omega_skew + ((1 - np.cos(theta)) / theta) * omega_skew_sq) / theta if theta > 1e-6 else np.eye(3)
            p = (np.eye(3) * theta + (1 - np.cos(theta)) * omega_skew + (theta - np.sin(theta)) * omega_skew_sq) @ v / (theta**2) if theta > 1e-6 else v

        # Create transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = p
        return T

    def forward_kinematics(self, thetalist, M):
        """
        Compute forward kinematics using product of exponentials
        thetalist: list of joint angles
        M: home configuration of end-effector
        """
        if len(thetalist) != self.N:
            raise ValueError("Number of joint angles must match number of joints")

        T = M.copy()
        for i in range(self.N-1, -1, -1):
            # Create twist matrix
            twist = np.zeros((4, 4))
            omega = self.screw_axes[0:3, i]
            v = self.screw_axes[3:6, i]

            twist[0:3, 0:3] = np.array([[0, -omega[2], omega[1]],
                                       [omega[2], 0, -omega[0]],
                                       [-omega[1], omega[0], 0]])
            twist[0:3, 3] = v

            # Apply transformation
            exp_twist = self.matrix_exp6(thetalist[i] * twist)
            T = exp_twist @ T

        return T
```

## Inverse Kinematics

Inverse kinematics (IK) is the reverse problem: given a desired end-effector pose, find the joint angles that achieve it. This is more challenging than forward kinematics and may have multiple solutions or no solution.

### Analytical Inverse Kinematics

For simple robots, analytical solutions exist:

```python
import numpy as np
import math


class AnalyticalIK2D:
    """Analytical inverse kinematics for a 2D planar 2-link manipulator"""

    def __init__(self, link_lengths):
        """
        Initialize with link lengths
        link_lengths: [L1, L2] - lengths of the two links
        """
        self.L1, self.L2 = link_lengths

    def solve_ik(self, x, y):
        """
        Solve inverse kinematics for 2D planar 2-link manipulator
        Returns two possible solutions (elbow up and elbow down)
        """
        # Check if the target is reachable
        distance = np.sqrt(x**2 + y**2)
        if distance > (self.L1 + self.L2):
            raise ValueError("Target is out of reach")
        if distance < abs(self.L1 - self.L2):
            raise ValueError("Target is inside the workspace")

        # Calculate angle for joint 2
        cos_theta2 = (x**2 + y**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        sin_theta2 = np.sqrt(1 - cos_theta2**2)

        theta2 = np.arctan2(sin_theta2, cos_theta2)  # Elbow up solution
        theta2_alt = np.arctan2(-sin_theta2, cos_theta2)  # Elbow down solution

        # Calculate angle for joint 1
        k1 = self.L1 + self.L2 * cos_theta2
        k2 = self.L2 * sin_theta2
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        # Alternative solution
        k1_alt = self.L1 + self.L2 * cos_theta2_alt
        k2_alt = self.L2 * sin_theta2_alt
        theta1_alt = np.arctan2(y, x) - np.arctan2(k2_alt, k1_alt)

        return [(theta1, theta2), (theta1_alt, theta2_alt)]


class AnalyticalIK3D:
    """Analytical inverse kinematics for a 6-DOF manipulator (simplified PUMA-like)"""

    def __init__(self, dh_params):
        """
        Initialize with DH parameters for a 6-DOF manipulator
        This is a simplified example for a specific robot configuration
        """
        self.dh_params = dh_params

    def solve_ik(self, target_position, target_orientation):
        """
        Solve inverse kinematics for a 6-DOF manipulator
        This is a simplified implementation for a specific robot type
        """
        x, y, z = target_position

        # For a simplified 6-DOF robot with spherical wrist, we can decouple
        # position and orientation kinematics

        # 1. Position: Calculate wrist center position
        # (Assuming spherical wrist offset)
        wrist_offset = 0.1  # Example wrist offset
        wrist_x = x - wrist_offset * target_orientation[0, 2]
        wrist_y = y - wrist_offset * target_orientation[1, 2]
        wrist_z = z - wrist_offset * target_orientation[2, 2]

        # 2. Calculate first three joints for wrist position (like 3-DOF arm)
        # Using the 2D solution approach but in 3D space
        r = np.sqrt(wrist_x**2 + wrist_y**2)
        alpha = np.arctan2(wrist_y, wrist_x)  # Joint 1

        # Now solve 2D problem in the rz plane
        d = wrist_z  # Height

        # This is a simplified version - full solution would involve
        # more complex geometric relationships
        solutions = []

        # For demonstration, return a placeholder solution
        # In a real implementation, you would solve the full geometric problem
        for i in range(2):  # Two possible arm configurations
            sol = [0.0] * 6  # 6 joint angles
            sol[0] = alpha  # Joint 1
            # Calculate remaining joints based on geometry
            # (This would require full geometric analysis)
            solutions.append(sol)

        return solutions


# Example usage
def main():
    # Example for 2D planar manipulator
    ik_2d = AnalyticalIK2D([1.0, 1.0])  # Two links of length 1.0

    try:
        solutions = ik_2d.solve_ik(1.0, 1.0)
        print(f"Target: (1.0, 1.0)")
        for i, (theta1, theta2) in enumerate(solutions):
            print(f"Solution {i+1}: Joint 1 = {np.degrees(theta1):.2f}°, Joint 2 = {np.degrees(theta2):.2f}°")
    except ValueError as e:
        print(f"Error: {e}")

    # Verify solution by running forward kinematics
    fk = ForwardKinematics([
        DHParameter(a=1.0, alpha=0, d=0, theta=0),
        DHParameter(a=1.0, alpha=0, d=0, theta=0)
    ])

    # Use first solution
    if solutions:
        theta1, theta2 = solutions[0]
        position, orientation = fk.get_end_effector_pose([theta1, theta2])
        print(f"FK verification - Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")


if __name__ == '__main__':
    main()
```

### Numerical Inverse Kinematics

For complex robots without analytical solutions, numerical methods are used:

```python
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


class NumericalIK:
    def __init__(self, robot_model):
        """
        Initialize with a robot model
        robot_model: object with forward_kinematics method
        """
        self.robot_model = robot_model

    def objective_function(self, joint_angles, target_pose):
        """
        Objective function to minimize
        joint_angles: current joint angles
        target_pose: desired end-effector pose [x, y, z, rx, ry, rz]
        """
        # Calculate current end-effector pose
        current_pose = self.robot_model.forward_kinematics(joint_angles)

        # Calculate position error
        pos_error = np.linalg.norm(current_pose[:3] - target_pose[:3])

        # Calculate orientation error
        current_rot = R.from_matrix(current_pose[3:].reshape(3, 3))
        target_rot = R.from_matrix(target_pose[3:].reshape(3, 3))
        rot_error = np.linalg.norm(current_rot.as_rotvec() - target_rot.as_rotvec())

        # Combined error
        total_error = pos_error + 0.1 * rot_error  # Weight orientation less

        return total_error

    def solve_ik(self, target_pose, initial_guess, joint_limits=None):
        """
        Solve inverse kinematics using numerical optimization
        target_pose: desired end-effector pose [x, y, z, r, p, y] or transformation matrix
        initial_guess: starting joint angles
        joint_limits: list of (min, max) for each joint
        """
        if joint_limits is None:
            joint_limits = [(-np.pi, np.pi)] * len(initial_guess)

        # Define bounds
        bounds = []
        for lim in joint_limits:
            bounds.append((lim[0], lim[1]))

        # Optimize
        result = minimize(
            self.objective_function,
            initial_guess,
            args=(target_pose,),
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success:
            return result.x
        else:
            raise RuntimeError(f"IK solution failed: {result.message}")


# Example using a simple robot model
class SimpleRobotModel:
    def __init__(self):
        # For this example, we'll use a simple model
        pass

    def forward_kinematics(self, joint_angles):
        """
        Simplified forward kinematics returning [x, y, z, r, p, y]
        This is a placeholder - in reality, you'd implement proper FK
        """
        # For a 3-DOF planar manipulator
        if len(joint_angles) >= 3:
            L1, L2, L3 = 1.0, 1.0, 0.5  # Link lengths
            theta1, theta2, theta3 = joint_angles[:3]

            x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3)
            y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2) + L3 * np.sin(theta1 + theta2 + theta3)
            z = 0  # Planar robot

            # Simple orientation (end-effector angle)
            end_effector_angle = theta1 + theta2 + theta3

            # Return position and orientation as Euler angles
            return np.array([x, y, z, 0, 0, end_effector_angle])
        else:
            return np.zeros(6)


def main():
    # Create robot model and IK solver
    robot_model = SimpleRobotModel()
    ik_solver = NumericalIK(robot_model)

    # Define target pose [x, y, z, roll, pitch, yaw]
    target_pose = np.array([1.5, 1.0, 0.0, 0.0, 0.0, np.pi/4])

    # Initial guess
    initial_guess = [0.0, 0.0, 0.0]

    # Joint limits (for example, ±170 degrees)
    joint_limits = [(-np.pi*0.95, np.pi*0.95)] * 3

    try:
        solution = ik_solver.solve_ik(target_pose, initial_guess, joint_limits)
        print(f"IK Solution found: {np.degrees(solution)} degrees")

        # Verify solution
        final_pose = robot_model.forward_kinematics(solution)
        print(f"Final pose: {final_pose}")
        print(f"Target pose: {target_pose}")
        print(f"Position error: {np.linalg.norm(final_pose[:3] - target_pose[:3]):.6f}")

    except RuntimeError as e:
        print(f"IK Error: {e}")


if __name__ == '__main__':
    main()
```

## Robot Dynamics

Robot dynamics deals with the forces and torques required to generate motion. Understanding dynamics is crucial for controlling robots with precision and efficiency.

### Newton-Euler Formulation

The Newton-Euler method calculates forces and moments acting on each link:

```python
import numpy as np


class NewtonEulerDynamics:
    def __init__(self, link_masses, link_lengths, link_coms):
        """
        Initialize dynamics model using Newton-Euler formulation
        link_masses: list of masses for each link
        link_lengths: list of lengths for each link
        link_coms: list of center of mass positions for each link
        """
        self.masses = np.array(link_masses)
        self.lengths = np.array(link_lengths)
        self.coms = np.array(link_coms)  # Center of mass positions relative to joint
        self.n = len(link_masses)

    def forward_dynamics(self, joint_positions, joint_velocities, joint_accelerations, joint_torques):
        """
        Calculate joint accelerations given torques (forward dynamics)
        """
        # This is a simplified implementation
        # Full Newton-Euler would require detailed kinematic relationships

        # Calculate mass matrix (simplified as diagonal)
        mass_matrix = np.diag(self.masses * self.coms**2)  # Simplified inertia approximation

        # Add Coriolis and centrifugal terms (simplified)
        coriolis_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # Simplified Coriolis term
                    coriolis_matrix[i, j] = self.masses[i] * self.coms[i] * self.coms[j] * joint_velocities[j]

        # Gravity terms
        gravity = 9.81
        gravity_terms = self.masses * gravity * np.sin(joint_positions)

        # Forward dynamics: M(q)q_ddot + C(q,q_dot)q_dot + g(q) = τ
        # Solve for q_ddot
        rhs = joint_torques - (coriolis_matrix @ joint_velocities) - gravity_terms
        joint_accelerations = np.linalg.solve(mass_matrix, rhs)

        return joint_accelerations

    def inverse_dynamics(self, joint_positions, joint_velocities, joint_accelerations):
        """
        Calculate required torques given motion (inverse dynamics)
        """
        # Calculate mass matrix
        mass_matrix = np.diag(self.masses * self.coms**2)  # Simplified

        # Calculate Coriolis matrix
        coriolis_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    coriolis_matrix[i, j] = self.masses[i] * self.coms[i] * self.coms[j] * joint_velocities[j]

        # Calculate gravity terms
        gravity = 9.81
        gravity_terms = self.masses * gravity * np.sin(joint_positions)

        # Calculate required torques
        # τ = M(q)q_ddot + C(q,q_dot)q_dot + g(q)
        joint_torques = (mass_matrix @ joint_accelerations +
                        coriolis_matrix @ joint_velocities +
                        gravity_terms)

        return joint_torques


# Example usage
def main():
    # Create a simple 2-DOF manipulator
    link_masses = [1.0, 0.8]  # Mass of each link in kg
    link_lengths = [1.0, 0.8]  # Length of each link in m
    link_coms = [0.5, 0.4]     # Center of mass position in m

    dynamics = NewtonEulerDynamics(link_masses, link_lengths, link_coms)

    # Example motion
    joint_positions = np.array([np.pi/4, np.pi/6])      # Joint angles
    joint_velocities = np.array([0.1, 0.05])            # Joint velocities
    joint_accelerations = np.array([0.01, 0.005])       # Joint accelerations

    # Calculate required torques (inverse dynamics)
    required_torques = dynamics.inverse_dynamics(joint_positions, joint_velocities, joint_accelerations)
    print(f"Required torques: {required_torques} Nm")

    # Calculate motion given torques (forward dynamics)
    applied_torques = required_torques  # Use same torques for verification
    resulting_accelerations = dynamics.forward_dynamics(
        joint_positions, joint_velocities, joint_accelerations, applied_torques
    )
    print(f"Resulting accelerations: {resulting_accelerations} rad/s²")


if __name__ == '__main__':
    main()
```

### Lagrangian Formulation

The Lagrangian method is another approach to robot dynamics:

```python
import numpy as np
from scipy.integrate import solve_ivp


class LagrangianDynamics:
    def __init__(self, robot_params):
        """
        Initialize dynamics model using Lagrangian formulation
        robot_params: Dictionary containing robot parameters
        """
        self.params = robot_params

    def mass_matrix(self, q):
        """
        Calculate mass matrix H(q)
        q: joint positions
        """
        # For a 2-DOF planar manipulator
        # H = [[h11, h12], [h12, h22]]

        m1, m2 = self.params['m1'], self.params['m2']
        l1, l2 = self.params['l1'], self.params['l2']
        lc1, lc2 = self.params['lc1'], self.params['lc2']
        I1, I2 = self.params['I1'], self.params['I2']

        q1, q2 = q[0], q[1]

        # Elements of mass matrix
        h11 = (m1 + m2) * lc1**2 + m2 * l1**2 + I1 + I2 + 2*m2*l1*lc2*np.cos(q2)
        h12 = m2*lc2**2 + I2 + m2*l1*lc2*np.cos(q2)
        h21 = h12  # Symmetric
        h22 = m2*lc2**2 + I2

        H = np.array([[h11, h12], [h21, h22]])
        return H

    def coriolis_matrix(self, q, q_dot):
        """
        Calculate Coriolis matrix C(q, q_dot)
        q: joint positions
        q_dot: joint velocities
        """
        m2 = self.params['m2']
        l1, l2 = self.params['l1'], self.params['l2']
        lc2 = self.params['lc2']
        q1, q2 = q[0], q[1]
        q1_dot, q2_dot = q_dot[0], q_dot[1]

        c11 = -2*m2*l1*lc2*np.sin(q2)*q2_dot
        c12 = -m2*l1*lc2*np.sin(q2)*q2_dot
        c21 = m2*l1*lc2*np.sin(q2)*q1_dot
        c22 = 0

        C = np.array([[c11, c12], [c21, c22]])
        return C

    def gravity_vector(self, q):
        """
        Calculate gravity vector g(q)
        q: joint positions
        """
        m1, m2 = self.params['m1'], self.params['m2']
        lc1, lc2 = self.params['lc1'], self.params['lc2']
        g = self.params['g']

        q1, q2 = q[0], q[1]

        g1 = (m1*lc1 + m2*l1)*g*np.cos(q1) + m2*lc2*g*np.cos(q1 + q2)
        g2 = m2*lc2*g*np.cos(q1 + q2)

        G = np.array([g1, g2])
        return G

    def inverse_dynamics(self, q, q_dot, q_ddot):
        """
        Calculate required torques using inverse dynamics
        q: joint positions
        q_dot: joint velocities
        q_ddot: joint accelerations
        """
        H = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)

        # τ = H(q)q_ddot + C(q,q_dot)q_dot + G(q)
        tau = H @ q_ddot + C @ q_dot + G
        return tau

    def forward_dynamics(self, q, q_dot, tau):
        """
        Calculate joint accelerations using forward dynamics
        q: joint positions
        q_dot: joint velocities
        tau: applied torques
        """
        H = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)

        # H(q)q_ddot = τ - C(q,q_dot)q_dot - G(q)
        q_ddot = np.linalg.solve(H, tau - C @ q_dot - G)
        return q_ddot

    def simulate_motion(self, initial_conditions, torques_func, t_span, t_eval):
        """
        Simulate robot motion given torque profile
        initial_conditions: [q0, q_dot0] initial positions and velocities
        torques_func: function that returns torques at time t
        t_span: (t_start, t_end)
        t_eval: time points to evaluate
        """
        def dynamics_func(t, y):
            """
            RHS of the differential equation for solve_ivp
            y = [q, q_dot] state vector
            """
            n = len(y) // 2
            q = y[:n]
            q_dot = y[n:]

            # Get torques at current time
            tau = torques_func(t)

            # Calculate accelerations
            q_ddot = self.forward_dynamics(q, q_dot, tau)

            # Return derivatives [q_dot, q_ddot]
            return np.concatenate([q_dot, q_ddot])

        # Initial state [q0, q_dot0]
        y0 = np.concatenate([initial_conditions[0], initial_conditions[1]])

        # Solve the differential equation
        solution = solve_ivp(
            dynamics_func,
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45'
        )

        return solution


# Example usage
def main():
    # Define robot parameters for a 2-DOF planar manipulator
    robot_params = {
        'm1': 2.0,   # Mass of link 1 (kg)
        'm2': 1.5,   # Mass of link 2 (kg)
        'l1': 1.0,   # Length of link 1 (m)
        'l2': 0.8,   # Length of link 2 (m)
        'lc1': 0.5,  # Distance to COM of link 1 (m)
        'lc2': 0.4,  # Distance to COM of link 2 (m)
        'I1': 0.2,   # Inertia of link 1 (kg*m²)
        'I2': 0.1,   # Inertia of link 2 (kg*m²)
        'g': 9.81    # Gravity (m/s²)
    }

    dynamics = LagrangianDynamics(robot_params)

    # Example: Calculate torques for a specific motion
    q = np.array([np.pi/4, np.pi/6])      # Joint positions
    q_dot = np.array([0.1, 0.05])         # Joint velocities
    q_ddot = np.array([0.01, 0.005])      # Joint accelerations

    required_torques = dynamics.inverse_dynamics(q, q_dot, q_ddot)
    print(f"Required torques: {required_torques} Nm")

    # Example: Forward dynamics (find accelerations for given torques)
    applied_torques = np.array([2.0, 1.0])  # Applied torques
    resulting_accelerations = dynamics.forward_dynamics(q, q_dot, applied_torques)
    print(f"Resulting accelerations: {resulting_accelerations} rad/s²")


if __name__ == '__main__':
    main()
```

## Trajectory Generation

Trajectory generation creates smooth, feasible paths for robot motion:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


class TrajectoryGenerator:
    def __init__(self):
        pass

    def cubic_polynomial_trajectory(self, q_start, q_end, t_start, t_end, qd_start=0, qd_end=0):
        """
        Generate cubic polynomial trajectory
        q_start, q_end: start and end positions
        t_start, t_end: start and end times
        qd_start, qd_end: start and end velocities (default 0)
        """
        dt = t_end - t_start

        # Coefficients for cubic polynomial: q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # Using boundary conditions:
        # q(t_start) = q_start, q(t_end) = q_end
        # q_dot(t_start) = qd_start, q_dot(t_end) = qd_end

        a0 = q_start
        a1 = qd_start
        a2 = (3/dt**2) * (q_end - q_start) - (2/dt) * qd_start - (1/dt) * qd_end
        a3 = (-2/dt**3) * (q_end - q_start) + (1/dt**2) * (qd_start + qd_end)

        def trajectory(t):
            t_rel = t - t_start
            q = a0 + a1*t_rel + a2*t_rel**2 + a3*t_rel**3
            qd = a1 + 2*a2*t_rel + 3*a3*t_rel**2
            qdd = 2*a2 + 6*a3*t_rel
            return q, qd, qdd

        return trajectory

    def quintic_polynomial_trajectory(self, q_start, q_end, t_start, t_end,
                                    qd_start=0, qd_end=0, qdd_start=0, qdd_end=0):
        """
        Generate quintic polynomial trajectory
        More smooth with continuous acceleration
        """
        dt = t_end - t_start

        # Coefficients for quintic polynomial: q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        a0 = q_start
        a1 = qd_start
        a2 = qdd_start / 2
        a3 = (20*q_end - 20*q_start - (8*qd_end + 12*qd_start)*dt - (3*qdd_start - qdd_end)*dt**2) / (2*dt**3)
        a4 = (30*q_start - 30*q_end + (14*qd_end + 16*qd_start)*dt + (3*qdd_start - 2*qdd_end)*dt**2) / (2*dt**4)
        a5 = (12*q_end - 12*q_start - 6*(qd_end + qd_start)*dt - (qdd_start - qdd_end)*dt**2) / (2*dt**5)

        def trajectory(t):
            t_rel = t - t_start
            q = a0 + a1*t_rel + a2*t_rel**2 + a3*t_rel**3 + a4*t_rel**4 + a5*t_rel**5
            qd = a1 + 2*a2*t_rel + 3*a3*t_rel**2 + 4*a4*t_rel**3 + 5*a5*t_rel**4
            qdd = 2*a2 + 6*a3*t_rel + 12*a4*t_rel**2 + 20*a5*t_rel**3
            return q, qd, qdd

        return trajectory

    def trapezoidal_trajectory(self, q_start, q_end, max_vel, max_acc):
        """
        Generate trapezoidal velocity profile
        q_start, q_end: start and end positions
        max_vel: maximum velocity
        max_acc: maximum acceleration
        """
        total_distance = q_end - q_start
        direction = 1 if total_distance > 0 else -1
        total_distance = abs(total_distance)

        # Calculate time for acceleration phase to reach max velocity
        acc_time = max_vel / max_acc

        # Distance covered during acceleration
        acc_distance = 0.5 * max_acc * acc_time**2

        # Check if we can reach max velocity
        if 2 * acc_distance >= total_distance:
            # Triangle profile - can't reach max velocity
            new_max_vel = np.sqrt(max_acc * total_distance)
            acc_time = new_max_vel / max_acc
            acc_distance = 0.5 * new_max_vel * acc_time
            const_time = 0
        else:
            # Trapezoidal profile
            const_distance = total_distance - 2 * acc_distance
            const_time = const_distance / max_vel

        total_time = 2 * acc_time + const_time

        def trajectory(t):
            # Phase 1: Acceleration
            if 0 <= t <= acc_time:
                q = q_start + direction * (0.5 * max_acc * t**2)
                qd = direction * max_acc * t
                qdd = direction * max_acc
            # Phase 2: Constant velocity
            elif acc_time < t <= acc_time + const_time:
                q = q_start + direction * (acc_distance + max_vel * (t - acc_time))
                qd = direction * max_vel
                qdd = 0
            # Phase 3: Deceleration
            elif acc_time + const_time < t <= total_time:
                t_dec = t - (acc_time + const_time)
                q = q_start + direction * (acc_distance + const_time * max_vel +
                                          max_vel * t_dec - 0.5 * max_acc * t_dec**2)
                qd = direction * (max_vel - max_acc * t_dec)
                qdd = -direction * max_acc
            else:
                q = q_end
                qd = 0
                qdd = 0

            return q, qd, qdd

        return trajectory, total_time

    def multi_dof_trajectory(self, waypoints, times, max_vel=1.0, max_acc=1.0):
        """
        Generate trajectory through multiple waypoints for multi-DOF robot
        waypoints: list of joint configurations [q1, q2, ..., qn]
        times: list of times for each waypoint
        """
        n_dof = len(waypoints[0])
        n_waypoints = len(waypoints)

        trajectories = []

        for i in range(n_dof):
            # Extract joint positions for this DOF
            joint_positions = [waypoint[i] for waypoint in waypoints]

            # Create spline for this joint
            spline = CubicSpline(times, joint_positions)
            trajectories.append(spline)

        def multi_dof_traj(t):
            positions = [traj(t) for traj in trajectories]
            velocities = [traj.derivative()(t) for traj in trajectories]
            accelerations = [traj.derivative().derivative()(t) for traj in trajectories]

            return np.array(positions), np.array(velocities), np.array(accelerations)

        return multi_dof_traj


# Example usage and visualization
def main():
    traj_gen = TrajectoryGenerator()

    # Example 1: Cubic polynomial trajectory
    print("=== Cubic Polynomial Trajectory ===")
    cubic_traj = traj_gen.cubic_polynomial_trajectory(
        q_start=0, q_end=2, t_start=0, t_end=4, qd_start=0, qd_end=0
    )

    t = np.linspace(0, 4, 100)
    q_cubic, qd_cubic, qdd_cubic = [], [], []
    for ti in t:
        qi, qdi, qddi = cubic_traj(ti)
        q_cubic.append(qi)
        qd_cubic.append(qdi)
        qdd_cubic.append(qddi)

    q_cubic = np.array(q_cubic)
    qd_cubic = np.array(qd_cubic)
    qdd_cubic = np.array(qdd_cubic)

    print(f"Cubic trajectory - Start: pos={q_cubic[0]:.3f}, vel={qd_cubic[0]:.3f}")
    print(f"Cubic trajectory - End: pos={q_cubic[-1]:.3f}, vel={qd_cubic[-1]:.3f}")

    # Example 2: Quintic polynomial trajectory
    print("\n=== Quintic Polynomial Trajectory ===")
    quintic_traj = traj_gen.quintic_polynomial_trajectory(
        q_start=0, q_end=2, t_start=0, t_end=4, qd_start=0, qd_end=0, qdd_start=0, qdd_end=0
    )

    q_quintic, qd_quintic, qdd_quintic = [], [], []
    for ti in t:
        qi, qdi, qddi = quintic_traj(ti)
        q_quintic.append(qi)
        qd_quintic.append(qdi)
        qdd_quintic.append(qddi)

    q_quintic = np.array(q_quintic)
    qd_quintic = np.array(qd_quintic)
    qdd_quintic = np.array(qdd_quintic)

    print(f"Quintic trajectory - Start: pos={q_quintic[0]:.3f}, vel={qd_quintic[0]:.3f}, acc={qdd_quintic[0]:.3f}")
    print(f"Quintic trajectory - End: pos={q_quintic[-1]:.3f}, vel={qd_quintic[-1]:.3f}, acc={qdd_quintic[-1]:.3f}")

    # Example 3: Trapezoidal trajectory
    print("\n=== Trapezoidal Trajectory ===")
    trap_traj, total_time = traj_gen.trapezoidal_trajectory(
        q_start=0, q_end=5, max_vel=2, max_acc=1
    )

    t_trap = np.linspace(0, total_time, 100)
    q_trap, qd_trap, qdd_trap = [], [], []
    for ti in t_trap:
        qi, qdi, qddi = trap_traj(ti)
        q_trap.append(qi)
        qd_trap.append(qdi)
        qdd_trap.append(qddi)

    q_trap = np.array(q_trap)
    qd_trap = np.array(qd_trap)
    qdd_trap = np.array(qdd_trap)

    print(f"Trapezoidal trajectory - Total time: {total_time:.3f}s")
    print(f"Trapezoidal trajectory - Start: pos={q_trap[0]:.3f}, vel={qd_trap[0]:.3f}")
    print(f"Trapezoidal trajectory - End: pos={q_trap[-1]:.3f}, vel={qd_trap[-1]:.3f}")

    # Visualization
    plt.figure(figsize=(15, 10))

    # Position subplot
    plt.subplot(3, 1, 1)
    plt.plot(t, q_cubic, label='Cubic', linewidth=2)
    plt.plot(t, q_quintic, label='Quintic', linewidth=2)
    plt.plot(t_trap, q_trap, label='Trapezoidal', linewidth=2)
    plt.title('Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)

    # Velocity subplot
    plt.subplot(3, 1, 2)
    plt.plot(t, qd_cubic, label='Cubic', linewidth=2)
    plt.plot(t, qd_quintic, label='Quintic', linewidth=2)
    plt.plot(t_trap, qd_trap, label='Trapezoidal', linewidth=2)
    plt.title('Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)

    # Acceleration subplot
    plt.subplot(3, 1, 3)
    plt.plot(t, qdd_cubic, label='Cubic', linewidth=2)
    plt.plot(t, qdd_quintic, label='Quintic', linewidth=2)
    plt.plot(t_trap, qdd_trap, label='Trapezoidal', linewidth=2)
    plt.title('Acceleration vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
```

## ROS 2 Integration for Kinematics and Dynamics

### Joint Trajectory Controller

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from builtin_interfaces.msg import Duration
import numpy as np


class JointTrajectoryController(Node):
    def __init__(self):
        super().__init__('joint_trajectory_controller')

        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publish desired joint positions
        self.joint_cmd_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        # Create action server for trajectory execution
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'follow_joint_trajectory',
            self.execute_trajectory
        )

        # Store current joint states
        self.current_joint_positions = {}
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # Example joint names

        # Trajectory execution parameters
        self.control_rate = 100  # Hz
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)

        self.get_logger().info('Joint trajectory controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for name, position in zip(msg.name, msg.position):
            self.current_joint_positions[name] = position

    def execute_trajectory(self, goal_handle):
        """Execute a joint trajectory goal"""
        self.get_logger().info('Executing trajectory...')

        trajectory = goal_handle.request.trajectory
        points = trajectory.points
        joint_names = trajectory.joint_names

        # Execute trajectory point by point
        for i, point in enumerate(points):
            # Calculate time to reach this point
            if i > 0:
                dt = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                prev_time = points[i-1].time_from_start.sec + points[i-1].time_from_start.nanosec * 1e-9
                sleep_time = dt - prev_time
            else:
                sleep_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9

            # Create joint state message
            joint_state = JointState()
            joint_state.name = joint_names
            joint_state.position = point.positions
            joint_state.velocity = point.velocities if point.velocities else [0.0] * len(point.positions)
            joint_state.effort = point.effort if point.effort else [0.0] * len(point.positions)

            # Publish the command
            self.joint_cmd_pub.publish(joint_state)

            # Wait for the specified time (in a real system, you'd use a more sophisticated approach)
            self.get_logger().info(f'Published trajectory point {i+1}/{len(points)}')

            # Check for preemption
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Trajectory execution canceled')
                return FollowJointTrajectory.Result()

        # Complete successfully
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        self.get_logger().info('Trajectory execution completed successfully')
        return result

    def control_loop(self):
        """Main control loop for smooth trajectory following"""
        # In a real implementation, this would interpolate between trajectory points
        # and publish commands at the control rate
        pass


def main(args=None):
    rclpy.init(args=args)
    controller = JointTrajectoryController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Hands-On Lab: Kinematic Control System

### Objective
Create a complete kinematic control system that solves inverse kinematics and generates smooth trajectories for a robotic arm.

### Prerequisites
- Completed Chapter 1-7
- ROS 2 Humble with Gazebo installed
- Basic understanding of robot kinematics

### Steps

1. **Create a kinematic control package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python kinematic_control_lab --dependencies rclpy sensor_msgs trajectory_msgs control_msgs geometry_msgs numpy scipy matplotlib
   ```

2. **Create the main kinematic control node** (`kinematic_control_lab/kinematic_control_lab/kinematic_control_node.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import JointState
   from geometry_msgs.msg import Pose, Point, Quaternion
   from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
   from std_msgs.msg import Header
   from builtin_interfaces.msg import Duration
   import numpy as np
   from scipy.spatial.transform import Rotation as R
   import time


   class DHParameter:
       def __init__(self, a, alpha, d, theta):
           self.a = a
           self.alpha = alpha
           self.d = d
           self.theta = theta

       def get_transformation_matrix(self):
           sa = np.sin(self.alpha)
           ca = np.cos(self.alpha)
           st = np.sin(self.theta)
           ct = np.cos(self.theta)

           T = np.array([
               [ct, -st*ca, st*sa, self.a*ct],
               [st, ct*ca, -ct*sa, self.a*st],
               [0, sa, ca, self.d],
               [0, 0, 0, 1]
           ])
           return T


   class KinematicControlNode(Node):
       def __init__(self):
           super().__init__('kinematic_control_node')

           # Publisher for joint commands
           self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)

           # Publisher for trajectory commands
           self.trajectory_pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)

           # Subscriber for target poses
           self.target_pose_sub = self.create_subscription(
               Pose,
               '/target_pose',
               self.target_pose_callback,
               10
           )

           # Initialize robot kinematics (6-DOF manipulator example)
           self.dh_params = [
               DHParameter(a=0, alpha=np.pi/2, d=0.5, theta=0),    # Joint 1
               DHParameter(a=0.5, alpha=0, d=0, theta=0),          # Joint 2
               DHParameter(a=0.4, alpha=0, d=0, theta=0),          # Joint 3
               DHParameter(a=0, alpha=np.pi/2, d=0.3, theta=0),    # Joint 4
               DHParameter(a=0, alpha=-np.pi/2, d=0, theta=0),     # Joint 5
               DHParameter(a=0, alpha=0, d=0.2, theta=0)           # Joint 6
           ]

           # Joint limits (in radians)
           self.joint_limits = [
               (-np.pi, np.pi),      # Joint 1
               (-np.pi/2, np.pi/2),  # Joint 2
               (-np.pi/2, np.pi/2),  # Joint 3
               (-np.pi, np.pi),      # Joint 4
               (-np.pi/2, np.pi/2),  # Joint 5
               (-np.pi, np.pi)       # Joint 6
           ]

           # Current joint positions
           self.current_joints = [0.0] * 6

           # Trajectory generation
           self.traj_gen = TrajectoryGenerator()

           self.get_logger().info('Kinematic control node initialized')

       def forward_kinematics(self, joint_angles):
           """Calculate forward kinematics"""
           if len(joint_angles) != len(self.dh_params):
               raise ValueError("Number of joint angles must match number of joints")

           # Update theta values in DH parameters
           dh_params = []
           for i, (param, angle) in enumerate(zip(self.dh_params, joint_angles)):
               new_param = DHParameter(param.a, param.alpha, param.d, angle + param.theta)
               dh_params.append(new_param)

           # Calculate cumulative transformation
           T_total = np.eye(4)
           for param in dh_params:
               T = param.get_transformation_matrix()
               T_total = T_total @ T

           # Extract position and orientation
           position = T_total[0:3, 3]
           orientation_matrix = T_total[0:3, 0:3]
           rotation = R.from_matrix(orientation_matrix)
           quaternion = rotation.as_quat()

           return position, quaternion

       def inverse_kinematics(self, target_position, target_orientation, initial_guess=None):
           """Solve inverse kinematics using numerical method"""
           if initial_guess is None:
               initial_guess = [0.0] * 6

           def objective_function(joint_angles):
               pos, quat = self.forward_kinematics(joint_angles)
               pos_error = np.linalg.norm(pos - target_position)
               # Simplified orientation error
               orient_error = 1 - np.abs(np.dot(quat, target_orientation))
               return pos_error + 0.1 * orient_error

           # Use scipy.optimize to solve IK
           from scipy.optimize import minimize

           # Apply joint limits
           bounds = self.joint_limits

           result = minimize(
               objective_function,
               initial_guess,
               method='L-BFGS-B',
               bounds=bounds
           )

           if result.success:
               return result.x
           else:
               self.get_logger().error(f'IK solution failed: {result.message}')
               return None

       def target_pose_callback(self, msg):
           """Handle target pose requests"""
           # Extract target position and orientation from message
           target_pos = np.array([msg.position.x, msg.position.y, msg.position.z])

           # Convert quaternion to array
           target_orient = np.array([
               msg.orientation.x,
               msg.orientation.y,
               msg.orientation.z,
               msg.orientation.w
           ])

           # Solve inverse kinematics
           joint_solution = self.inverse_kinematics(target_pos, target_orient)

           if joint_solution is not None:
               self.get_logger().info(f'IK Solution: {np.degrees(joint_solution)} degrees')

               # Generate smooth trajectory to the target
               self.execute_smooth_trajectory(joint_solution)
           else:
               self.get_logger().error('No IK solution found')

       def execute_smooth_trajectory(self, target_joints):
           """Execute smooth trajectory to target joint positions"""
           # Get current joint positions
           current_joints = self.current_joints

           # Create trajectory points
           n_points = 50  # Number of intermediate points
           trajectory = JointTrajectory()
           trajectory.joint_names = [f'joint_{i+1}' for i in range(6)]

           # Generate intermediate points
           for i in range(n_points + 1):
               fraction = i / n_points
               intermediate_joints = (1 - fraction) * np.array(current_joints) + fraction * np.array(target_joints)

               point = JointTrajectoryPoint()
               point.positions = intermediate_joints.tolist()

               # Calculate velocities (simple linear interpolation)
               if i > 0:
                   dt = 0.1  # Time step
                   prev_joints = (1 - (i-1)/n_points) * np.array(current_joints) + ((i-1)/n_points) * np.array(target_joints)
                   velocities = (intermediate_joints - prev_joints) / dt
                   point.velocities = velocities.tolist()
               else:
                   point.velocities = [0.0] * 6

               # Set time from start
               point.time_from_start = Duration(sec=0, nanosec=int(i * 100000000))  # 0.1s per point

               trajectory.points.append(point)

           # Update current joints
           self.current_joints = target_joints.tolist()

           # Publish trajectory
           trajectory.header.stamp = self.get_clock().now().to_msg()
           trajectory.header.frame_id = 'base_link'
           self.trajectory_pub.publish(trajectory)

           self.get_logger().info(f'Published trajectory with {len(trajectory.points)} points')

       def move_to_joint_positions(self, joint_positions):
           """Move robot to specific joint positions"""
           if len(joint_positions) != 6:
               self.get_logger().error('Need exactly 6 joint positions')
               return

           # Create trajectory message
           trajectory = JointTrajectory()
           trajectory.joint_names = [f'joint_{i+1}' for i in range(6)]

           # Single point trajectory
           point = JointTrajectoryPoint()
           point.positions = joint_positions
           point.velocities = [0.0] * 6
           point.time_from_start = Duration(sec=2, nanosec=0)  # 2 seconds to reach

           trajectory.points.append(point)
           trajectory.header.stamp = self.get_clock().now().to_msg()
           trajectory.header.frame_id = 'base_link'

           self.trajectory_pub.publish(trajectory)
           self.current_joints = joint_positions


   class TrajectoryGenerator:
       def __init__(self):
           pass

       def cubic_polynomial_trajectory(self, q_start, q_end, t_start, t_end, qd_start=0, qd_end=0):
           """Generate cubic polynomial trajectory for single joint"""
           dt = t_end - t_start

           a0 = q_start
           a1 = qd_start
           a2 = (3/dt**2) * (q_end - q_start) - (2/dt) * qd_start - (1/dt) * qd_end
           a3 = (-2/dt**3) * (q_end - q_start) + (1/dt**2) * (qd_start + qd_end)

           def trajectory(t):
               t_rel = t - t_start
               q = a0 + a1*t_rel + a2*t_rel**2 + a3*t_rel**3
               qd = a1 + 2*a2*t_rel + 3*a3*t_rel**2
               return q, qd

           return trajectory


   def main(args=None):
       rclpy.init(args=args)
       kinematic_control_node = KinematicControlNode()

       # Example: Move to a specific joint configuration after startup
       def startup_timer_callback():
           kinematic_control_node.get_logger().info('Moving to initial position...')
           initial_pos = [0.1, 0.2, 0.0, 0.0, 0.1, 0.0]
           kinematic_control_node.move_to_joint_positions(initial_pos)
           startup_timer.cancel()

       startup_timer = kinematic_control_node.create_timer(2.0, startup_timer_callback)

       try:
           rclpy.spin(kinematic_control_node)
       except KeyboardInterrupt:
           pass
       finally:
           kinematic_control_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`kinematic_control_lab/launch/kinematic_control.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation (Gazebo) clock if true'
       )

       # Kinematic control node
       kinematic_control_node = Node(
           package='kinematic_control_lab',
           executable='kinematic_control_node',
           name='kinematic_control_node',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
           output='screen'
       )

       return LaunchDescription([
           use_sim_time,
           kinematic_control_node
       ])
   ```

4. **Update setup.py**:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'kinematic_control_lab'

   setup(
       name=package_name,
       version='0.0.0',
       packages=find_packages(exclude=['test']),
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Kinematic control lab for robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'kinematic_control_node = kinematic_control_lab.kinematic_control_node:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select kinematic_control_lab
   source install/setup.bash
   ```

6. **Run the kinematic control system**:
   ```bash
   ros2 launch kinematic_control_lab kinematic_control.launch.py
   ```

### Expected Results
- The system should solve inverse kinematics for target end-effector poses
- Smooth trajectories should be generated and published
- Joint positions should be updated to move the robot to target configurations
- The system should handle joint limits appropriately

### Troubleshooting Tips
- Verify joint names match your robot's configuration
- Check that DH parameters match your robot's physical structure
- Ensure joint limits are appropriate for your robot
- Monitor the logs for IK solution status and trajectory execution

## Summary

In this chapter, we've covered the fundamental concepts of robot kinematics and dynamics, including forward and inverse kinematics, dynamics modeling, and trajectory generation. We've implemented practical examples of each concept and created a complete kinematic control system.

The hands-on lab provided experience with creating a system that solves inverse kinematics and generates smooth trajectories for robot motion. This foundation is essential for more advanced robotic capabilities like motion planning, control, and interaction with the environment, which we'll explore in the upcoming chapters.