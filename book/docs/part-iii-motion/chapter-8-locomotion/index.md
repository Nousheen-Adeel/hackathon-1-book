---
title: Chapter 8 - Locomotion and Balance Control
sidebar_position: 2
---

# Chapter 8: Locomotion and Balance Control

## Learning Goals

- Understand bipedal locomotion principles
- Learn balance control and stabilization
- Master walking pattern generation
- Implement balance control algorithms
- Generate walking patterns for humanoid robots
- Simulate stable locomotion in various terrains

## Introduction to Bipedal Locomotion

Bipedal locomotion is one of the most challenging problems in robotics, requiring sophisticated control algorithms to maintain balance while moving on two legs. Unlike wheeled or tracked robots, bipedal robots must manage their center of mass, coordinate multiple joints, and adapt to changing terrain conditions.

### Challenges of Bipedal Locomotion

1. **Dynamic Balance**: Maintaining stability during motion
2. **Underactuation**: Fewer actuators than degrees of freedom in some phases
3. **Impact Dynamics**: Managing foot-ground collisions
4. **Terrain Adaptation**: Adjusting to uneven surfaces
5. **Energy Efficiency**: Minimizing power consumption during walking

### Locomotion Patterns

Bipedal robots can use various walking patterns:

- **Static Walking**: Center of mass always over support polygon
- **Dynamic Walking**: Center of mass may move outside support polygon
- **Passive Dynamic Walking**: Using gravity and momentum for efficiency
- **ZMP-Based Walking**: Zero Moment Point control for stability

## Balance Control Fundamentals

### Center of Mass and Stability

The center of mass (CoM) is crucial for balance control. A humanoid robot is stable when its CoM projection falls within the support polygon defined by its feet.

```python
import numpy as np
import matplotlib.pyplot as plt


class BalanceController:
    def __init__(self, robot_mass=75.0, com_height=0.85):
        """
        Initialize balance controller
        robot_mass: Total mass of the robot in kg
        com_height: Height of center of mass in meters
        """
        self.robot_mass = robot_mass
        self.com_height = com_height
        self.gravity = 9.81
        self.com_position = np.array([0.0, 0.0, com_height])  # x, y, z
        self.com_velocity = np.array([0.0, 0.0, 0.0])
        self.com_acceleration = np.array([0.0, 0.0, 0.0])

    def calculate_support_polygon(self, left_foot_pos, right_foot_pos, foot_width=0.15):
        """
        Calculate support polygon from foot positions
        left_foot_pos, right_foot_pos: 3D positions of feet [x, y, z]
        foot_width: Width of foot in meters
        """
        # For simplicity, assume rectangular feet
        # In practice, this would be more complex based on contact points
        support_points = [
            [left_foot_pos[0] - foot_width/2, left_foot_pos[1] - foot_width/2],
            [left_foot_pos[0] + foot_width/2, left_foot_pos[1] - foot_width/2],
            [right_foot_pos[0] + foot_width/2, right_foot_pos[1] + foot_width/2],
            [right_foot_pos[0] - foot_width/2, right_foot_pos[1] + foot_width/2]
        ]

        return np.array(support_points)

    def is_stable(self, com_proj, support_polygon):
        """
        Check if CoM projection is within support polygon
        com_proj: 2D projection of CoM [x, y]
        support_polygon: 2D vertices of support polygon
        """
        # Use ray casting algorithm to check if point is in polygon
        x, y = com_proj
        n = len(support_polygon)
        inside = False

        p1x, p1y = support_polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = support_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def calculate_zmp(self, com_pos, com_vel, com_acc):
        """
        Calculate Zero Moment Point (ZMP)
        com_pos: Center of mass position [x, y, z]
        com_vel: Center of mass velocity [x, y, z]
        com_acc: Center of mass acceleration [x, y, z]
        """
        x, y, z = com_pos
        x_dot, y_dot, z_dot = com_vel
        x_ddot, y_ddot, z_ddot = com_acc

        # ZMP calculation (simplified, assuming constant CoM height)
        zmp_x = x - (z - self.com_height) * x_ddot / self.gravity
        zmp_y = y - (z - self.com_height) * y_ddot / self.gravity

        return np.array([zmp_x, zmp_y])


# Example usage
def main():
    controller = BalanceController()

    # Example foot positions
    left_foot = np.array([0.1, 0.1, 0.0])
    right_foot = np.array([-0.1, -0.1, 0.0])

    # Calculate support polygon
    support_poly = controller.calculate_support_polygon(left_foot, right_foot)

    # Example CoM position
    com_pos = np.array([0.0, 0.0, 0.85])
    com_vel = np.array([0.01, -0.02, 0.0])
    com_acc = np.array([0.05, -0.03, 0.0])

    # Calculate ZMP
    zmp = controller.calculate_zmp(com_pos, com_vel, com_acc)

    # Check stability
    com_proj = com_pos[:2]  # Project CoM to 2D
    stable = controller.is_stable(com_proj, support_poly)

    print(f"Support polygon: {support_poly}")
    print(f"CoM projection: {com_proj}")
    print(f"ZMP: {zmp}")
    print(f"Stable: {stable}")

    # Visualization
    plt.figure(figsize=(10, 8))

    # Plot support polygon
    support_poly_closed = np.vstack([support_poly, support_poly[0]])  # Close the polygon
    plt.plot(support_poly_closed[:, 0], support_poly_closed[:, 1], 'b-', linewidth=2, label='Support Polygon')
    plt.fill(support_poly[:, 0], support_poly[:, 1], alpha=0.3, color='blue')

    # Plot CoM and ZMP
    plt.plot(com_proj[0], com_proj[1], 'ro', markersize=10, label='CoM Projection')
    plt.plot(zmp[0], zmp[1], 'gs', markersize=10, label='ZMP')

    plt.title('Balance Control: Support Polygon, CoM, and ZMP')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
```

## Inverted Pendulum Model

The inverted pendulum is a fundamental model for understanding balance control:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class InvertedPendulum:
    def __init__(self, mass=1.0, length=1.0, gravity=9.81):
        """
        Initialize inverted pendulum model
        mass: Mass of the pendulum bob
        length: Length of the pendulum
        gravity: Gravitational acceleration
        """
        self.mass = mass
        self.length = length
        self.gravity = gravity
        self.moment_of_inertia = mass * length**2

    def dynamics(self, t, state, control_input=0):
        """
        Equations of motion for inverted pendulum
        state: [theta, theta_dot] where theta is angle from vertical
        control_input: Torque applied at the pivot
        """
        theta, theta_dot = state

        # Nonlinear dynamics of inverted pendulum
        # theta_ddot = (g*sin(theta) - control_input/(m*l^2)) / l
        theta_ddot = (self.gravity * np.sin(theta) - control_input / (self.mass * self.length**2)) / self.length

        return [theta_dot, theta_ddot]

    def linearize(self):
        """
        Linearize the system around the upright position (theta = 0)
        Returns A and B matrices for linear system: x_dot = Ax + Bu
        """
        # For small angles, sin(theta) ≈ theta
        # State: x = [theta, theta_dot]
        # x_dot = [theta_dot, (g/l)*theta - 1/(m*l^2)*u]
        A = np.array([
            [0, 1],
            [self.gravity/self.length, 0]
        ])
        B = np.array([
            [0],
            [-1/(self.mass * self.length**2)]
        ])

        return A, B

    def simulate(self, initial_state, control_func, t_span, t_eval):
        """
        Simulate the inverted pendulum
        initial_state: [theta, theta_dot] at t=0
        control_func: Function that returns control input given (t, state)
        """
        def dynamics_with_control(t, state):
            control_input = control_func(t, state)
            return self.dynamics(t, state, control_input)

        solution = solve_ivp(
            dynamics_with_control,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='RK45'
        )

        return solution


class BalanceControllerInvertedPendulum:
    def __init__(self, pendulum):
        """
        Initialize balance controller based on inverted pendulum model
        """
        self.pendulum = pendulum
        self.A, self.B = pendulum.linearize()

        # Design controller using pole placement
        # Choose desired closed-loop poles
        desired_poles = [-2, -3]  # Faster response
        self.K = self.compute_feedback_gain()

    def compute_feedback_gain(self):
        """
        Compute state feedback gain using pole placement
        For 2-state system: control = -K * state
        """
        # For this simple example, we'll use a simple approach
        # In practice, you'd use more sophisticated methods like LQR or pole placement
        # Using Ackermann's formula or scipy's place function

        # Simple PD controller approach
        # For system x_dot = Ax + Bu, u = -Kx
        # We want eigenvalues of (A-BK) to be at desired locations
        # For now, return a simple gain matrix
        K = np.array([10.0, 2.0])  # PD controller gains
        return K

    def control_law(self, state):
        """
        Compute control input based on current state
        state: [theta, theta_dot]
        """
        control = -self.K @ state
        return control[0] if hasattr(control, '__len__') else control


# Example usage
def main():
    # Create inverted pendulum model
    pendulum = InvertedPendulum(mass=10.0, length=0.85)  # Approximate leg length

    # Create balance controller
    controller = BalanceControllerInvertedPendulum(pendulum)

    # Simulate with balance control
    initial_state = [0.1, 0.0]  # Small initial angle
    t_span = (0, 5)
    t_eval = np.linspace(0, 5, 500)

    def control_func(t, state):
        return controller.control_law(state)

    solution = pendulum.simulate(initial_state, control_func, t_span, t_eval)

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(solution.t, solution.y[0], 'b-', linewidth=2, label='Angle (rad)')
    plt.title('Inverted Pendulum with Balance Control')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(solution.t, solution.y[1], 'r-', linewidth=2, label='Angular Velocity (rad/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Final angle: {solution.y[0][-1]:.4f} rad")
    print(f"Final angular velocity: {solution.y[1][-1]:.4f} rad/s")


if __name__ == '__main__':
    main()
```

## Walking Pattern Generation

### ZMP-Based Walking Controller

Zero Moment Point (ZMP) based walking is a widely used approach for humanoid locomotion:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumtrapz


class ZMPPatternGenerator:
    def __init__(self, sampling_time=0.01, com_height=0.85, gravity=9.81):
        """
        Initialize ZMP-based walking pattern generator
        sampling_time: Time step for pattern generation
        com_height: Center of mass height
        gravity: Gravitational acceleration
        """
        self.dt = sampling_time
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(self.gravity / self.com_height)  # Natural frequency

    def generate_com_trajectory_from_zmp(self, zmp_trajectory):
        """
        Generate CoM trajectory from ZMP reference using the linear inverted pendulum model
        x_com_ddot - omega^2 * x_com = -omega^2 * x_zmp
        """
        t = np.arange(0, len(zmp_trajectory)) * self.dt

        # For the linear inverted pendulum model:
        # x_com_ddot - omega^2 * x_com = -omega^2 * x_zmp
        # This is a second-order linear ODE that can be solved analytically

        x_zmp = zmp_trajectory[:, 0]  # X component of ZMP
        y_zmp = zmp_trajectory[:, 1]  # Y component of ZMP

        # Solve for CoM trajectory (simplified approach)
        # The solution to x_ddot - omega^2*x = -omega^2*zmp is:
        # x_com(t) = x_h(t) + x_p(t)
        # where x_h is homogeneous solution and x_p is particular solution

        # For demonstration, use a simplified approach with pre-computed trajectory
        # In practice, you'd solve the full differential equation

        # Generate CoM trajectory using preview control approach
        com_x = self._solve_com_trajectory(x_zmp)
        com_y = self._solve_com_trajectory(y_zmp)

        # Calculate velocities and accelerations by differentiation
        com_x_vel = np.gradient(com_x, self.dt)
        com_y_vel = np.gradient(com_y, self.dt)
        com_x_acc = np.gradient(com_x_vel, self.dt)
        com_y_acc = np.gradient(com_y_vel, self.dt)

        # Create full trajectory with all states
        com_trajectory = np.column_stack([com_x, com_y, np.full_like(com_x, self.com_height)])
        com_velocity = np.column_stack([com_x_vel, com_y_vel, np.zeros_like(com_x_vel)])
        com_acceleration = np.column_stack([com_x_acc, com_y_acc, np.zeros_like(com_x_acc)])

        return com_trajectory, com_velocity, com_acceleration

    def _solve_com_trajectory(self, zmp_ref):
        """Helper function to solve CoM trajectory for one dimension"""
        # This is a simplified implementation
        # In practice, you'd use preview control with a longer preview window

        # For now, return a smoothed version of the ZMP reference
        # Apply a low-pass filter to make CoM trajectory smooth
        b, a = signal.butter(2, 0.1, 'low', fs=1/self.dt)
        com_pos = signal.filtfilt(b, a, zmp_ref)
        return com_pos

    def generate_footprint_pattern(self, step_length=0.3, step_width=0.2, n_steps=10):
        """
        Generate a simple walking footprint pattern
        step_length: Forward step length
        step_width: Lateral distance between feet
        n_steps: Number of steps to generate
        """
        footsteps = []

        # Start with left foot at origin
        left_pos = np.array([0.0, step_width/2, 0.0])
        right_pos = np.array([0.0, -step_width/2, 0.0])

        for i in range(n_steps):
            # Odd steps: move right foot
            if i % 2 == 1:
                right_pos[0] += step_length
                right_pos[1] = (-1)**i * step_width/2
                footsteps.append(('right', right_pos.copy()))
            # Even steps: move left foot
            else:
                left_pos[0] += step_length
                left_pos[1] = (-1)**(i+1) * step_width/2
                footsteps.append(('left', left_pos.copy()))

        return footsteps

    def generate_zmp_trajectory(self, footsteps, double_support_time=0.1, dt=0.01):
        """
        Generate ZMP trajectory based on footsteps
        footsteps: List of (foot_type, position) tuples
        double_support_time: Time spent in double support phase
        """
        # Calculate total time
        single_support_time = 1.0  # Time for single support phase
        total_time = len(footsteps) * (single_support_time + double_support_time)

        # Create time vector
        t = np.arange(0, total_time, dt)

        # Initialize ZMP trajectory
        zmp_trajectory = np.zeros((len(t), 2))  # x, y components

        # Generate ZMP pattern for each step
        for i, (foot_type, foot_pos) in enumerate(footsteps):
            step_start_time = i * (single_support_time + double_support_time)
            double_support_end = step_start_time + double_support_time
            step_end_time = step_start_time + single_support_time + double_support_time

            # Find time indices for this step
            start_idx = int(step_start_time / dt)
            double_end_idx = int(double_support_end / dt)
            end_idx = int(step_end_time / dt)

            if end_idx >= len(t):
                end_idx = len(t)

            # Double support phase: ZMP transitions between feet
            if double_end_idx > start_idx:
                for j in range(start_idx, min(double_end_idx, len(t))):
                    # Interpolate between previous foot position and current foot position
                    if i == 0:
                        # For first step, start from middle position
                        prev_pos = np.array([0.0, 0.0])
                    else:
                        prev_type, prev_pos = footsteps[i-1]
                        prev_pos = prev_pos[:2]

                    t_interp = (t[j] - step_start_time) / double_support_time
                    zmp_trajectory[j] = (1 - t_interp) * prev_pos + t_interp * foot_pos[:2]

            # Single support phase: ZMP stays at current foot
            if end_idx > double_end_idx:
                zmp_trajectory[double_end_idx:end_idx, 0] = foot_pos[0]
                zmp_trajectory[double_end_idx:end_idx, 1] = foot_pos[1]

        return t, zmp_trajectory


# Example usage
def main():
    # Create ZMP pattern generator
    zmp_gen = ZMPPatternGenerator()

    # Generate footsteps
    footsteps = zmp_gen.generate_footprint_pattern(step_length=0.3, step_width=0.2, n_steps=8)
    print(f"Generated {len(footsteps)} footsteps")

    # Display footsteps
    for i, (foot_type, pos) in enumerate(footsteps):
        print(f"Step {i+1}: {foot_type} foot at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # Generate ZMP trajectory
    t, zmp_trajectory = zmp_gen.generate_zmp_trajectory(footsteps, double_support_time=0.2)

    # Generate CoM trajectory from ZMP
    com_trajectory, com_velocity, com_acceleration = zmp_gen.generate_com_trajectory_from_zmp(zmp_trajectory)

    # Visualization
    plt.figure(figsize=(15, 10))

    # Plot ZMP and CoM trajectories
    plt.subplot(2, 2, 1)
    plt.plot(zmp_trajectory[:, 0], zmp_trajectory[:, 1], 'r-', linewidth=2, label='ZMP')
    plt.plot(com_trajectory[:, 0], com_trajectory[:, 1], 'b-', linewidth=2, label='CoM')

    # Mark footsteps
    left_x, left_y = [], []
    right_x, right_y = [], []
    for foot_type, pos in footsteps:
        if foot_type == 'left':
            left_x.append(pos[0])
            left_y.append(pos[1])
        else:
            right_x.append(pos[0])
            right_y.append(pos[1])

    plt.scatter(left_x, left_y, c='g', s=100, marker='^', label='Left Foot')
    plt.scatter(right_x, right_y, c='m', s=100, marker='v', label='Right Foot')

    plt.title('ZMP and CoM Trajectories')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Plot X trajectories over time
    plt.subplot(2, 2, 2)
    plt.plot(t, zmp_trajectory[:, 0], 'r-', linewidth=2, label='ZMP X')
    plt.plot(t, com_trajectory[:, 0], 'b-', linewidth=2, label='CoM X')
    plt.title('X Position over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    # Plot Y trajectories over time
    plt.subplot(2, 2, 3)
    plt.plot(t, zmp_trajectory[:, 1], 'r-', linewidth=2, label='ZMP Y')
    plt.plot(t, com_trajectory[:, 1], 'b-', linewidth=2, label='CoM Y')
    plt.title('Y Position over Time')
    plt.xlabel('Time (m)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    # Plot CoM height (constant)
    plt.subplot(2, 2, 4)
    plt.plot(t, com_trajectory[:, 2], 'g-', linewidth=2, label='CoM Z')
    plt.title('CoM Height (Constant)')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
```

### Walking Pattern with Capture Point

The Capture Point is a useful concept for balance control during walking:

```python
import numpy as np
import matplotlib.pyplot as plt


class CapturePointController:
    def __init__(self, com_height=0.85, gravity=9.81):
        """
        Initialize Capture Point controller
        com_height: Center of mass height
        gravity: Gravitational acceleration
        """
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate the capture point
        The capture point is where the robot should step to stop safely
        capture_point = com_pos + com_vel / omega
        """
        cp_x = com_pos[0] + com_vel[0] / self.omega
        cp_y = com_pos[1] + com_vel[1] / self.omega
        return np.array([cp_x, cp_y])

    def should_step(self, com_pos, com_vel, foot_positions, step_threshold=0.1):
        """
        Determine if a step is needed based on capture point
        foot_positions: Dictionary with 'left' and 'right' foot positions
        step_threshold: Distance threshold for stepping
        """
        capture_point = self.calculate_capture_point(com_pos, com_vel)

        # Find the support polygon (area between feet)
        all_foot_pos = np.array([pos for pos in foot_positions.values()])
        support_center_x = np.mean(all_foot_pos[:, 0])
        support_center_y = np.mean(all_foot_pos[:, 1])

        # Calculate distance from capture point to support center
        cp_distance = np.sqrt((capture_point[0] - support_center_x)**2 +
                              (capture_point[1] - support_center_y)**2)

        return cp_distance > step_threshold, capture_point

    def generate_step_location(self, capture_point, current_foot_pos, step_size=0.3):
        """
        Generate appropriate step location based on capture point
        """
        # Step toward the capture point, but with a reasonable step size
        direction = capture_point - current_foot_pos[:2]
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0:
            # Normalize direction and scale to step size
            step_direction = direction / direction_norm
            step_location = current_foot_pos[:2] + step_direction * min(step_size, direction_norm)
        else:
            step_location = current_foot_pos[:2]

        # Add small Z component for foot lift
        step_location_full = np.array([step_location[0], step_location[1], 0.0])

        return step_location_full


class WalkingController:
    def __init__(self, com_height=0.85):
        self.com_height = com_height
        self.capture_controller = CapturePointController(com_height)

        # Robot state
        self.com_pos = np.array([0.0, 0.0, com_height])
        self.com_vel = np.array([0.0, 0.0, 0.0])
        self.left_foot_pos = np.array([0.0, 0.1, 0.0])
        self.right_foot_pos = np.array([0.0, -0.1, 0.0])

        # Walking parameters
        self.step_length = 0.3
        self.step_width = 0.2
        self.step_height = 0.05  # Foot lift height

    def update_balance(self, dt):
        """
        Update balance based on current state
        """
        # Calculate if we need to step
        foot_positions = {
            'left': self.left_foot_pos,
            'right': self.right_foot_pos
        }

        should_step, capture_point = self.capture_controller.should_step(
            self.com_pos, self.com_vel, foot_positions
        )

        if should_step:
            print(f"Balance: Need to step! Capture point at ({capture_point[0]:.3f}, {capture_point[1]:.3f})")
            # In a real system, this would trigger a stepping motion
            return True, capture_point

        return False, capture_point

    def generate_foot_trajectory(self, start_pos, end_pos, step_height=0.05, steps=20):
        """
        Generate smooth foot trajectory from start to end position
        """
        t = np.linspace(0, 1, steps)

        # Linear interpolation for x, y
        x_traj = start_pos[0] + (end_pos[0] - start_pos[0]) * t
        y_traj = start_pos[1] + (end_pos[1] - start_pos[1]) * t

        # Sinusoidal trajectory for z (foot lift)
        z_lift = start_pos[2] + (end_pos[2] - start_pos[2]) * t
        # Add foot lift in the middle of the step
        z_traj = z_lift + step_height * np.sin(np.pi * t)

        return np.column_stack([x_traj, y_traj, z_traj])


# Example usage
def main():
    walker = WalkingController()

    # Simulate a walking scenario
    time_points = []
    com_positions = []
    step_needed = []

    # Simulate forward walking motion
    dt = 0.01
    simulation_time = 10.0
    t = 0

    while t < simulation_time:
        # Simulate forward motion (push CoM forward)
        walker.com_vel[0] = 0.2  # Push forward
        walker.com_pos += walker.com_vel * dt

        # Add some small disturbances
        walker.com_pos[1] += np.random.normal(0, 0.001) * dt  # Small lateral disturbance
        walker.com_vel[1] += np.random.normal(0, 0.01) * dt   # Small velocity disturbance

        # Check balance
        needs_step, cp = walker.update_balance(dt)
        step_needed.append(needs_step)
        time_points.append(t)
        com_positions.append(walker.com_pos.copy())

        t += dt

    # Convert to arrays
    time_points = np.array(time_points)
    com_positions = np.array(com_positions)
    step_needed = np.array(step_needed)

    print(f"Simulation completed. Steps needed: {np.sum(step_needed)} out of {len(step_needed)} time steps")

    # Visualization
    plt.figure(figsize=(15, 10))

    # Plot CoM trajectory
    plt.subplot(2, 2, 1)
    plt.plot(com_positions[:, 0], com_positions[:, 1], 'b-', linewidth=2, label='CoM Trajectory')
    plt.scatter(com_positions[step_needed, 0], com_positions[step_needed, 1],
               c='red', s=10, alpha=0.5, label='Step Needed')
    plt.title('Center of Mass Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Plot X position over time
    plt.subplot(2, 2, 2)
    plt.plot(time_points, com_positions[:, 0], 'b-', linewidth=2, label='X Position')
    plt.title('CoM X Position over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.legend()
    plt.grid(True)

    # Plot Y position over time
    plt.subplot(2, 2, 3)
    plt.plot(time_points, com_positions[:, 1], 'r-', linewidth=2, label='Y Position')
    plt.title('CoM Y Position over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)

    # Plot when steps are needed
    plt.subplot(2, 2, 4)
    plt.plot(time_points, step_needed.astype(int), 'g-', linewidth=2, label='Step Needed')
    plt.title('Step Needed Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Step Needed (1=Yes, 0=No)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
```

## Balance Control Algorithms

### Linear Quadratic Regulator (LQR) for Balance

LQR is a powerful control technique for balance control:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are


class LQRBalanceController:
    def __init__(self, com_height=0.85, mass=75.0, gravity=9.81):
        """
        Initialize LQR balance controller
        com_height: Center of mass height
        mass: Robot mass
        gravity: Gravitational acceleration
        """
        self.com_height = com_height
        self.mass = mass
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

        # State: [x, x_dot, y, y_dot] where x, y are CoM positions and velocities
        # Control input: [F_x, F_y] forces applied at CoM
        self.A = np.array([
            [0, 1, 0, 0],           # dx/dt = x_dot
            [self.omega**2, 0, 0, 0], # dx_dot/dt = omega^2 * x
            [0, 0, 0, 1],           # dy/dt = y_dot
            [0, 0, self.omega**2, 0]  # dy_dot/dt = omega^2 * y
        ])

        # Control input matrix (force -> acceleration)
        self.B = np.array([
            [0, 0],
            [1/(self.mass * self.com_height), 0],
            [0, 0],
            [0, 1/(self.mass * self.com_height)]
        ])

        # Design LQR controller
        self.Q = np.diag([100, 1, 100, 1])  # State cost matrix
        self.R = np.diag([1, 1])            # Control cost matrix
        self.K = self.compute_lqr_gain()

    def compute_lqr_gain(self):
        """
        Compute LQR gain matrix K such that u = -Kx
        """
        # Solve the continuous-time algebraic Riccati equation
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Compute the optimal gain
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K

    def control(self, state):
        """
        Compute control input for given state
        state: [x, x_dot, y, y_dot] - CoM position and velocity
        """
        # State feedback: u = -Kx
        control_input = -self.K @ state
        return control_input

    def get_force_control(self, com_pos, com_vel):
        """
        Get force control input given CoM position and velocity
        com_pos: [x, y] - CoM position
        com_vel: [x_dot, y_dot] - CoM velocity
        """
        state = np.array([com_pos[0], com_vel[0], com_pos[1], com_vel[1]])
        control_input = self.control(state)
        return control_input

    def simulate_balance(self, initial_state, simulation_time=10.0, dt=0.01):
        """
        Simulate balance control
        initial_state: [x0, x_dot0, y0, y_dot0]
        """
        t = np.arange(0, simulation_time, dt)
        states = np.zeros((len(t), 4))
        states[0] = initial_state
        controls = np.zeros((len(t), 2))

        for i in range(1, len(t)):
            # Get control input
            control_input = self.control(states[i-1])
            controls[i-1] = control_input

            # Update state: x_dot = Ax + Bu
            state_dot = self.A @ states[i-1] + self.B @ control_input
            states[i] = states[i-1] + state_dot * dt

        return t, states, controls


# Example usage
def main():
    # Create LQR balance controller
    controller = LQRBalanceController(com_height=0.85, mass=75.0)

    # Simulate with initial disturbance
    initial_state = np.array([0.05, 0.0, 0.02, 0.0])  # Small initial position errors
    t, states, controls = controller.simulate_balance(initial_state, simulation_time=5.0)

    # Extract components
    x_pos = states[:, 0]
    x_vel = states[:, 1]
    y_pos = states[:, 2]
    y_vel = states[:, 3]

    control_x = controls[:, 0]
    control_y = controls[:, 1]

    print(f"LQR Balance Controller - Initial state: {initial_state}")
    print(f"Final state after 5s: x={x_pos[-1]:.4f}, x_dot={x_vel[-1]:.4f}, y={y_pos[-1]:.4f}, y_dot={y_vel[-1]:.4f}")

    # Visualization
    plt.figure(figsize=(15, 12))

    # Plot CoM position
    plt.subplot(3, 2, 1)
    plt.plot(t, x_pos, 'b-', linewidth=2, label='X Position')
    plt.plot(t, y_pos, 'r-', linewidth=2, label='Y Position')
    plt.title('CoM Position over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    # Plot CoM velocity
    plt.subplot(3, 2, 2)
    plt.plot(t, x_vel, 'b-', linewidth=2, label='X Velocity')
    plt.plot(t, y_vel, 'r-', linewidth=2, label='Y Velocity')
    plt.title('CoM Velocity over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)

    # Phase plot (position vs velocity)
    plt.subplot(3, 2, 3)
    plt.plot(x_pos, x_vel, 'b-', linewidth=2, label='X Phase Plot')
    plt.title('X Phase Plot (Position vs Velocity)')
    plt.xlabel('Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(3, 2, 4)
    plt.plot(y_pos, y_vel, 'r-', linewidth=2, label='Y Phase Plot')
    plt.title('Y Phase Plot (Position vs Velocity)')
    plt.xlabel('Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Control inputs
    plt.subplot(3, 2, 5)
    plt.plot(t, control_x, 'b-', linewidth=2, label='X Force Control')
    plt.plot(t, control_y, 'r-', linewidth=2, label='Y Force Control')
    plt.title('Control Forces over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.grid(True)

    # State trajectory in 2D
    plt.subplot(3, 2, 6)
    plt.plot(x_pos, y_pos, 'g-', linewidth=2, label='CoM Trajectory')
    plt.plot(x_pos[0], y_pos[0], 'go', markersize=10, label='Start')
    plt.plot(x_pos[-1], y_pos[-1], 'ro', markersize=10, label='End')
    plt.title('CoM Trajectory in 2D')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
```

## ROS 2 Integration for Locomotion Control

### Complete Walking Controller Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
import numpy as np


class WalkingControllerNode(Node):
    def __init__(self):
        super().__init__('walking_controller')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.com_state_pub = self.create_publisher(Float64MultiArray, '/com_state', 10)
        self.zmp_pub = self.create_publisher(Float64MultiArray, '/zmp', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.foot_pressure_sub = self.create_subscription(
            Float64MultiArray,
            '/foot_pressure',
            self.foot_pressure_callback,
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        # Walking state
        self.current_joints = {}
        self.left_foot_pressure = [0.0, 0.0, 0.0, 0.0]  # 4 pressure sensors
        self.right_foot_pressure = [0.0, 0.0, 0.0, 0.0]

        # Initialize controllers
        self.balance_controller = LQRBalanceController()
        self.zmp_generator = ZMPPatternGenerator()
        self.capture_controller = CapturePointController()

        # Walking parameters
        self.com_height = 0.85
        self.walking_state = 'standing'  # standing, walking, stepping
        self.step_phase = 0.0  # 0.0 to 1.0, where 0.0 is start of step, 1.0 is end

        self.get_logger().info('Walking controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos

    def foot_pressure_callback(self, msg):
        """Update foot pressure sensor readings"""
        # Assuming first 4 values are left foot, next 4 are right foot
        if len(msg.data) >= 8:
            self.left_foot_pressure = msg.data[:4]
            self.right_foot_pressure = msg.data[4:8]

    def estimate_com_state(self):
        """
        Estimate center of mass position and velocity from joint states
        This is a simplified approach - in practice, you'd use forward kinematics
        """
        # For demonstration, return a simple estimate
        # In a real system, you'd calculate this from the full kinematic model
        com_pos = np.array([0.0, 0.0, self.com_height])
        com_vel = np.array([0.0, 0.0, 0.0])

        return com_pos, com_vel

    def calculate_zmp_from_pressure(self):
        """
        Calculate ZMP from foot pressure sensors
        """
        # Calculate ZMP from pressure distribution
        # Simplified calculation using pressure sensor locations
        sensor_positions = np.array([
            [-0.05, 0.05],   # front-left
            [0.05, 0.05],    # front-right
            [-0.05, -0.05],  # back-left
            [0.05, -0.05]    # back-right
        ])

        # Calculate left foot ZMP
        left_total_pressure = sum(self.left_foot_pressure)
        if left_total_pressure > 0:
            left_zmp_x = sum(p * pos[0] for p, pos in zip(self.left_foot_pressure, sensor_positions)) / left_total_pressure
            left_zmp_y = sum(p * pos[1] for p, pos in zip(self.left_foot_pressure, sensor_positions)) / left_total_pressure
            left_zmp = np.array([left_zmp_x, left_zmp_y])
        else:
            left_zmp = np.array([0.0, 0.0])

        # Calculate right foot ZMP
        right_total_pressure = sum(self.right_foot_pressure)
        if right_total_pressure > 0:
            right_zmp_x = sum(p * pos[0] for p, pos in zip(self.right_foot_pressure, sensor_positions)) / right_total_pressure
            right_zmp_y = sum(p * pos[1] for p, pos in zip(self.right_foot_pressure, sensor_positions)) / right_total_pressure
            right_zmp = np.array([right_zmp_x, right_zmp_y])
        else:
            right_zmp = np.array([0.0, 0.0])

        # Overall ZMP based on both feet
        total_pressure = left_total_pressure + right_total_pressure
        if total_pressure > 0:
            overall_zmp = (left_total_pressure * left_zmp + right_total_pressure * right_zmp) / total_pressure
        else:
            overall_zmp = np.array([0.0, 0.0])

        return overall_zmp

    def control_loop(self):
        """Main control loop"""
        # Estimate current CoM state
        com_pos, com_vel = self.estimate_com_state()

        # Calculate current ZMP
        current_zmp = self.calculate_zmp_from_pressure()

        # Publish CoM state
        com_msg = Float64MultiArray()
        com_msg.data = [com_pos[0], com_pos[1], com_pos[2], com_vel[0], com_vel[1], com_vel[2]]
        self.com_state_pub.publish(com_msg)

        # Publish ZMP
        zmp_msg = Float64MultiArray()
        zmp_msg.data = [current_zmp[0], current_zmp[1]]
        self.zmp_pub.publish(zmp_msg)

        # Balance control
        state = np.array([com_pos[0], com_vel[0], com_pos[1], com_vel[1]])
        control_forces = self.balance_controller.control(state)

        # For walking, we would also need to generate stepping patterns
        # and coordinate with joint controllers
        self.get_logger().info(f'Balance control: F_x={control_forces[0]:.3f} N, F_y={control_forces[1]:.3f} N')

        # Generate appropriate joint commands based on balance control
        # This would involve inverse kinematics and joint-level control
        # For now, just publish a placeholder
        joint_cmd = JointState()
        joint_cmd.name = [f'joint_{i}' for i in range(12)]  # Example joint names
        joint_cmd.position = [0.0] * 12  # Placeholder positions
        joint_cmd.velocity = [0.0] * 12
        joint_cmd.effort = [0.0] * 12

        # Add balance corrections to joint positions
        # This is where the high-level balance commands are translated to joint commands
        self.joint_cmd_pub.publish(joint_cmd)

        # Update walking state based on ZMP and other factors
        self.update_walking_state(current_zmp, state)

    def update_walking_state(self, current_zmp, com_state):
        """Update walking state based on ZMP and CoM state"""
        # Calculate capture point
        capture_point = self.capture_controller.calculate_capture_point(
            com_state[::2], com_state[1::2]  # pos and vel from state vector
        )

        # Check if we need to step
        # This would involve checking if capture point is outside support polygon
        # For now, just log the values
        self.get_logger().info(
            f'ZMP: ({current_zmp[0]:.3f}, {current_zmp[1]:.3f}), '
            f'Capture Point: ({capture_point[0]:.3f}, {capture_point[1]:.3f})'
        )


def main(args=None):
    rclpy.init(args=args)
    walking_controller = WalkingControllerNode()

    try:
        rclpy.spin(walking_controller)
    except KeyboardInterrupt:
        pass
    finally:
        walking_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Hands-On Lab: Bipedal Walking Controller

### Objective
Create a complete bipedal walking controller that implements balance control, ZMP-based walking patterns, and capture point stepping.

### Prerequisites
- Completed Chapter 1-8
- ROS 2 Humble with Gazebo installed
- Basic understanding of robot dynamics and control

### Steps

1. **Create a walking control package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python walking_control_lab --dependencies rclpy sensor_msgs geometry_msgs std_msgs numpy scipy matplotlib
   ```

2. **Create the main walking controller node** (`walking_control_lab/walking_control_lab/walking_controller_node.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import JointState
   from geometry_msgs.msg import Pose, Twist
   from std_msgs.msg import Float64MultiArray, Bool
   from builtin_interfaces.msg import Duration
   import numpy as np
   import time


   class ZMPPatternGenerator:
       def __init__(self, sampling_time=0.01, com_height=0.85, gravity=9.81):
           self.dt = sampling_time
           self.com_height = com_height
           self.gravity = gravity
           self.omega = np.sqrt(self.gravity / self.com_height)

       def generate_com_trajectory_from_zmp(self, zmp_trajectory):
           """Generate CoM trajectory from ZMP reference"""
           t = np.arange(0, len(zmp_trajectory)) * self.dt
           x_zmp = zmp_trajectory[:, 0]
           y_zmp = zmp_trajectory[:, 1]

           # Solve the differential equation: x_com_ddot - omega^2 * x_com = -omega^2 * x_zmp
           # Using numerical integration
           def integrate_zmp(zmp_ref):
               # Simple forward integration approach
               # In practice, use more sophisticated methods like preview control
               com_pos = np.zeros_like(zmp_ref)
               com_vel = np.zeros_like(zmp_ref)
               com_acc = np.zeros_like(zmp_ref)

               for i in range(1, len(zmp_ref)):
                   # x_com_ddot = omega^2 * (x_com - x_zmp)
                   if i > 1:
                       com_acc[i-1] = self.omega**2 * (com_pos[i-1] - zmp_ref[i-1])
                       com_vel[i] = com_vel[i-1] + com_acc[i-1] * self.dt
                       com_pos[i] = com_pos[i-1] + com_vel[i-1] * self.dt

               # Apply feedback to track ZMP
               for i in range(len(zmp_ref)):
                   com_pos[i] = zmp_ref[i] + com_vel[i] / self.omega

               return com_pos

           com_x = integrate_zmp(x_zmp)
           com_y = integrate_zmp(y_zmp)

           com_x_vel = np.gradient(com_x, self.dt)
           com_y_vel = np.gradient(com_y, self.dt)
           com_x_acc = np.gradient(com_x_vel, self.dt)
           com_y_acc = np.gradient(com_y_vel, self.dt)

           com_trajectory = np.column_stack([com_x, com_y, np.full_like(com_x, self.com_height)])
           com_velocity = np.column_stack([com_x_vel, com_y_vel, np.zeros_like(com_x_vel)])

           return com_trajectory, com_velocity

       def generate_footprint_pattern(self, step_length=0.3, step_width=0.2, n_steps=10):
           """Generate walking footprint pattern"""
           footsteps = []
           left_pos = np.array([0.0, step_width/2, 0.0])
           right_pos = np.array([0.0, -step_width/2, 0.0])

           for i in range(n_steps):
               if i % 2 == 1:
                   right_pos[0] += step_length
                   right_pos[1] = (-1)**i * step_width/2
                   footsteps.append(('right', right_pos.copy()))
               else:
                   left_pos[0] += step_length
                   left_pos[1] = (-1)**(i+1) * step_width/2
                   footsteps.append(('left', left_pos.copy()))

           return footsteps

       def generate_zmp_trajectory(self, footsteps, double_support_time=0.1):
           """Generate ZMP trajectory based on footsteps"""
           dt = self.dt
           single_support_time = 1.0
           total_time = len(footsteps) * (single_support_time + double_support_time)

           t = np.arange(0, total_time, dt)
           zmp_trajectory = np.zeros((len(t), 2))

           for i, (foot_type, foot_pos) in enumerate(footsteps):
               step_start_time = i * (single_support_time + double_support_time)
               double_support_end = step_start_time + double_support_time
               step_end_time = step_start_time + single_support_time + double_support_time

               start_idx = int(step_start_time / dt)
               double_end_idx = int(double_support_end / dt)
               end_idx = int(step_end_time / dt)

               if end_idx >= len(t):
                   end_idx = len(t)

               # Double support: interpolate between feet
               if double_end_idx > start_idx and i > 0:
                   prev_type, prev_pos = footsteps[i-1]
                   prev_pos_2d = prev_pos[:2]
                   for j in range(start_idx, min(double_end_idx, len(t))):
                       t_interp = min(1.0, (t[j] - step_start_time) / double_support_time)
                       zmp_trajectory[j] = (1 - t_interp) * prev_pos_2d + t_interp * foot_pos[:2]
               elif end_idx > double_end_idx:
                   # Single support: ZMP at foot position
                   zmp_trajectory[double_end_idx:end_idx, 0] = foot_pos[0]
                   zmp_trajectory[double_end_idx:end_idx, 1] = foot_pos[1]

           return t, zmp_trajectory


   class CapturePointController:
       def __init__(self, com_height=0.85, gravity=9.81):
           self.com_height = com_height
           self.gravity = gravity
           self.omega = np.sqrt(gravity / com_height)

       def calculate_capture_point(self, com_pos, com_vel):
           """Calculate capture point"""
           cp_x = com_pos[0] + com_vel[0] / self.omega
           cp_y = com_pos[1] + com_vel[1] / self.omega
           return np.array([cp_x, cp_y])


   class WalkingControllerNode(Node):
       def __init__(self):
           super().__init__('walking_controller_lab')

           # Publishers
           self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
           self.com_state_pub = self.create_publisher(Float64MultiArray, '/com_state', 10)
           self.zmp_pub = self.create_publisher(Float64MultiArray, '/zmp', 10)
           self.status_pub = self.create_publisher(Bool, '/walking_active', 10)

           # Subscribers
           self.joint_state_sub = self.create_subscription(
               JointState,
               '/joint_states',
               self.joint_state_callback,
               10
           )

           # Timer for control loop
           self.control_timer = self.create_timer(0.01, self.control_loop)

           # Initialize controllers
           self.zmp_generator = ZMPPatternGenerator(com_height=0.85)
           self.capture_controller = CapturePointController(com_height=0.85)

           # Walking state
           self.current_joints = {}
           self.com_height = 0.85
           self.walking_state = 'standing'
           self.step_count = 0
           self.balance_error = 0.0

           # Initialize with a simple walking pattern
           self.footsteps = self.zmp_generator.generate_footprint_pattern(
               step_length=0.3, step_width=0.2, n_steps=6
           )
           self.t_zmp, self.zmp_trajectory = self.zmp_generator.generate_zmp_trajectory(
               self.footsteps, double_support_time=0.2
           )
           self.com_trajectory, self.com_velocity = self.zmp_generator.generate_com_trajectory_from_zmp(
               self.zmp_trajectory
           )

           # Current trajectory index
           self.trajectory_idx = 0

           self.get_logger().info('Walking controller lab node initialized')

       def joint_state_callback(self, msg):
           """Update current joint positions"""
           for name, pos in zip(msg.name, msg.position):
               self.current_joints[name] = pos

       def estimate_com_state(self):
           """Estimate CoM state from current configuration"""
           # In a real system, this would use forward kinematics
           # For this example, use the precomputed trajectory
           if self.trajectory_idx < len(self.com_trajectory):
               com_pos = self.com_trajectory[self.trajectory_idx]
               com_vel = self.com_velocity[self.trajectory_idx]
               self.trajectory_idx += 1
               if self.trajectory_idx >= len(self.com_trajectory):
                   self.trajectory_idx = 0  # Loop
           else:
               com_pos = np.array([0.0, 0.0, self.com_height])
               com_vel = np.array([0.0, 0.0, 0.0])

           return com_pos, com_vel

       def calculate_balance_error(self, com_pos, zmp_pos):
           """Calculate balance error as distance between CoM and ZMP"""
           # Simple distance measure
           error = np.linalg.norm(com_pos[:2] - zmp_pos)
           return error

       def generate_joint_commands(self, com_command, foot_positions):
           """Generate joint commands to achieve CoM position"""
           # This is a simplified approach
           # In practice, this would involve full inverse kinematics
           # and whole-body control

           # For demonstration, return a simple pattern
           joint_cmd = JointState()
           joint_cmd.name = [
               'left_hip_roll', 'left_hip_pitch', 'left_hip_yaw',
               'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
               'right_hip_roll', 'right_hip_pitch', 'right_hip_yaw',
               'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
           ]

           # Simple mapping from CoM command to joint angles
           # In reality, this would use inverse kinematics
           base_angles = [0.0] * 12
           # Add balance corrections based on CoM error
           com_error_x = com_command[0]
           com_error_y = com_command[1]

           # Adjust hip and ankle angles based on CoM position
           base_angles[1] = -com_error_x * 0.5  # Left hip pitch
           base_angles[7] = -com_error_x * 0.5  # Right hip pitch
           base_angles[4] = com_error_y * 0.3   # Left ankle pitch
           base_angles[10] = com_error_y * 0.3  # Right ankle pitch

           joint_cmd.position = base_angles
           joint_cmd.velocity = [0.0] * 12
           joint_cmd.effort = [0.0] * 12

           return joint_cmd

       def control_loop(self):
           """Main control loop"""
           # Estimate current CoM state
           com_pos, com_vel = self.estimate_com_state()

           # Calculate current ZMP (from precomputed trajectory)
           if self.trajectory_idx < len(self.zmp_trajectory):
               current_zmp = self.zmp_trajectory[self.trajectory_idx-1] if self.trajectory_idx > 0 else np.array([0.0, 0.0])
           else:
               current_zmp = np.array([0.0, 0.0])

           # Calculate balance error
           self.balance_error = self.calculate_balance_error(com_pos, current_zmp)

           # Publish CoM state
           com_msg = Float64MultiArray()
           com_msg.data = [com_pos[0], com_pos[1], com_pos[2], com_vel[0], com_vel[1], com_vel[2]]
           self.com_state_pub.publish(com_msg)

           # Publish ZMP
           zmp_msg = Float64MultiArray()
           zmp_msg.data = [current_zmp[0], current_zmp[1]]
           self.zmp_pub.publish(zmp_msg)

           # Generate joint commands based on balance control
           com_command = [com_pos[0], com_pos[1], com_pos[2]]  # Use current CoM as command
           foot_positions = {}  # In a real system, you'd get actual foot positions
           joint_cmd = self.generate_joint_commands(com_command, foot_positions)

           # Publish joint commands
           self.joint_cmd_pub.publish(joint_cmd)

           # Publish status
           status_msg = Bool()
           status_msg.data = self.balance_error < 0.1  # Stable if error < 0.1m
           self.status_pub.publish(status_msg)

           # Log balance information
           self.get_logger().info(
               f'Balance Error: {self.balance_error:.3f}m, '
               f'CoM: ({com_pos[0]:.3f}, {com_pos[1]:.3f}), '
               f'ZMP: ({current_zmp[0]:.3f}, {current_zmp[1]:.3f})'
           )

           # Update walking state
           if self.balance_error > 0.2:
               self.walking_state = 'unstable'
           elif self.balance_error < 0.05:
               self.walking_state = 'stable'
           else:
               self.walking_state = 'walking'


   def main(args=None):
       rclpy.init(args=args)
       walking_controller = WalkingControllerNode()

       try:
           rclpy.spin(walking_controller)
       except KeyboardInterrupt:
           pass
       finally:
           walking_controller.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`walking_control_lab/launch/walking_control.launch.py`):
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

       # Walking controller node
       walking_controller_node = Node(
           package='walking_control_lab',
           executable='walking_controller_node',
           name='walking_controller_lab',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
           output='screen'
       )

       return LaunchDescription([
           use_sim_time,
           walking_controller_node
       ])
   ```

4. **Update setup.py**:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'walking_control_lab'

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
       description='Walking control lab for robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'walking_controller_node = walking_control_lab.walking_controller_node:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select walking_control_lab
   source install/setup.bash
   ```

6. **Run the walking control system**:
   ```bash
   ros2 launch walking_control_lab walking_control.launch.py
   ```

### Expected Results
- The system should maintain balance by tracking ZMP references
- Joint commands should be generated to maintain CoM stability
- Balance error should remain within acceptable limits
- The system should demonstrate basic walking patterns

### Troubleshooting Tips
- Ensure joint names match your robot's configuration
- Verify CoM height parameter matches your robot
- Check that ZMP references are feasible for your robot
- Monitor balance error to ensure stability

## Summary

In this chapter, we've explored the fundamental concepts of bipedal locomotion and balance control, including inverted pendulum models, ZMP-based walking, capture point control, and LQR balance controllers. We've implemented practical examples of each concept and created a complete walking control system.

The hands-on lab provided experience with creating a system that combines balance control, walking pattern generation, and joint-level control. This foundation is essential for advanced humanoid robotics applications and prepares us for more complex locomotion and control challenges in the upcoming chapters.