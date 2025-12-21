---
title: Chapter 4 - Sensor Integration and Data Processing
sidebar_position: 1
---

# Chapter 4: Sensor Integration and Data Processing

## Learning Goals

- Understand various robot sensors and their applications
- Learn to process sensor data streams
- Master sensor fusion techniques
- Integrate cameras, LIDAR, IMU, and other sensors
- Process and visualize sensor data streams
- Implement basic sensor fusion

## Introduction to Robot Sensors

Robots operate in complex, dynamic environments that require them to perceive and understand their surroundings. This perception is achieved through various sensors that provide information about the robot's internal state and external environment. Understanding how to integrate and process sensor data is fundamental to creating intelligent robotic systems.

### Sensor Categories

Robot sensors can be broadly categorized into:

1. **Proprioceptive Sensors**: Measure the robot's internal state (joint angles, motor currents, etc.)
2. **Exteroceptive Sensors**: Measure the external environment (cameras, LIDAR, etc.)
3. **Interoceptive Sensors**: Measure internal systems (temperature, battery level, etc.)

### Sensor Characteristics

When working with sensors, it's important to understand their key characteristics:

- **Resolution**: The smallest change a sensor can detect
- **Accuracy**: How close the measurement is to the true value
- **Precision**: How repeatable the measurements are
- **Range**: The minimum and maximum values the sensor can measure
- **Bandwidth**: The frequency range over which the sensor operates
- **Latency**: The delay between measurement and output
- **Noise**: Random variations in the measurement

## Common Robot Sensors

### Cameras

Cameras are among the most important sensors for robots, providing rich visual information about the environment. They can be categorized as:

- **Monocular Cameras**: Single camera providing 2D images
- **Stereo Cameras**: Two cameras providing depth information
- **RGB-D Cameras**: Provide both color and depth data
- **Fish-eye Cameras**: Provide wide-angle views

```python
# Example of camera data processing with ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image (example: edge detection)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Display the processed image
        cv2.imshow('Original', cv_image)
        cv2.imshow('Edges', edges)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    camera_processor = CameraProcessor()
    rclpy.spin(camera_processor)
    cv2.destroyAllWindows()
    camera_processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### LIDAR (Light Detection and Ranging)

LIDAR sensors provide accurate distance measurements by emitting laser pulses and measuring the time it takes for them to return. They are essential for mapping and navigation.

```python
# Example of LIDAR data processing
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt


class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.scan_data = None

    def scan_callback(self, msg):
        # Extract ranges from the scan message
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges (inf or nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        # Calculate statistics
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            max_distance = np.max(valid_ranges)
            avg_distance = np.mean(valid_ranges)

            self.get_logger().info(
                f'LIDAR: Min={min_distance:.2f}, Max={max_distance:.2f}, Avg={avg_distance:.2f}'
            )

        # Store for visualization
        self.scan_data = {
            'ranges': ranges,
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        }


def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LidarProcessor()
    rclpy.spin(lidar_processor)
    lidar_processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Inertial Measurement Units (IMU)

IMUs measure linear acceleration and angular velocity, providing information about the robot's motion and orientation. They typically contain accelerometers, gyroscopes, and sometimes magnetometers.

```python
# Example of IMU data processing
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np


class ImuProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Initialize variables for orientation estimation
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.linear_acceleration = np.array([0.0, 0.0, 0.0])

    def imu_callback(self, msg):
        # Extract orientation (if available)
        if msg.orientation_covariance[0] >= 0:  # Check if orientation is valid
            self.orientation = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])

        # Extract angular velocity
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Extract linear acceleration
        self.linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Log the data
        self.get_logger().info(
            f'IMU: AngVel=({self.angular_velocity[0]:.3f}, {self.angular_velocity[1]:.3f}, {self.angular_velocity[2]:.3f}), '
            f'LinAcc=({self.linear_acceleration[0]:.3f}, {self.linear_acceleration[1]:.3f}, {self.linear_acceleration[2]:.3f})'
        )


def main(args=None):
    rclpy.init(args=args)
    imu_processor = ImuProcessor()
    rclpy.spin(imu_processor)
    imu_processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Other Sensors

#### GPS (Global Positioning System)
- Provides absolute position information
- Limited accuracy in indoor environments
- Often used for outdoor navigation

#### Force/Torque Sensors
- Measure forces and torques applied to the robot
- Critical for manipulation tasks
- Used for compliant control

#### Temperature Sensors
- Monitor internal and external temperatures
- Important for system safety
- Used for thermal management

## Sensor Data Processing

### Time Synchronization

When working with multiple sensors, it's crucial to synchronize their data in time. ROS 2 provides message filters for this purpose:

```python
# Example of sensor synchronization
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import numpy as np


class SensorSynchronizer(Node):
    def __init__(self):
        super().__init__('sensor_synchronizer')

        # Create subscribers for different sensor topics
        image_sub = Subscriber(self, Image, '/camera/image_raw')
        scan_sub = Subscriber(self, LaserScan, '/scan')

        # Synchronize messages based on timestamps
        self.ts = ApproximateTimeSynchronizer(
            [image_sub, scan_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.ts.registerCallback(self.sync_callback)

        self.bridge = CvBridge()

    def sync_callback(self, image_msg, scan_msg):
        # Process synchronized sensor data
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        ranges = np.array(scan_msg.ranges)

        self.get_logger().info(f'Synchronized: Image at {image_msg.header.stamp.sec}, Scan at {scan_msg.header.stamp.sec}')


def main(args=None):
    rclpy.init(args=args)
    synchronizer = SensorSynchronizer()
    rclpy.spin(synchronizer)
    synchronizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Data Filtering

Sensor data often contains noise that needs to be filtered for reliable operation:

```python
# Example of sensor data filtering
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class SensorFilter:
    def __init__(self, cutoff_freq=10.0, sampling_freq=100.0):
        # Design a low-pass filter
        nyquist_freq = sampling_freq / 2.0
        normalized_cutoff = cutoff_freq / nyquist_freq

        # Create Butterworth filter
        self.b, self.a = signal.butter(4, normalized_cutoff, btype='low', analog=False)

        # Initialize filter state
        self.z = signal.lfilter_zi(self.b, self.a)

    def filter_data(self, new_sample):
        # Apply the filter to new data
        filtered, self.z = signal.lfilter(self.b, self.a, [new_sample], zi=self.z)
        return filtered[0]


# Example usage
def main():
    # Simulate noisy sensor data
    t = np.linspace(0, 1, 1000)
    true_signal = np.sin(2 * np.pi * 1 * t)  # 1 Hz signal
    noise = np.random.normal(0, 0.1, len(t))  # Add noise
    noisy_signal = true_signal + noise

    # Apply filtering
    sensor_filter = SensorFilter(cutoff_freq=5.0, sampling_freq=100.0)
    filtered_signal = np.zeros_like(noisy_signal)

    for i in range(len(noisy_signal)):
        filtered_signal[i] = sensor_filter.filter_data(noisy_signal[i])

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.plot(t, true_signal, label='True Signal', linewidth=2)
    plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.7)
    plt.plot(t, filtered_signal, label='Filtered Signal', linewidth=2)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Sensor Data Filtering Example')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
```

## Sensor Fusion

Sensor fusion combines data from multiple sensors to improve the accuracy and reliability of the robot's perception. Common fusion techniques include:

### Kalman Filtering

The Kalman filter is a mathematical method that uses a series of measurements observed over time to estimate unknown variables.

```python
# Simple Kalman Filter implementation for position estimation
import numpy as np


class KalmanFilter:
    def __init__(self, dt=0.1, process_noise=1.0, measurement_noise=1.0):
        # Time step
        self.dt = dt

        # State transition matrix (position and velocity)
        self.F = np.array([[1, dt],
                          [0, 1]])

        # Control matrix (not used in this example)
        self.B = np.array([[0.5 * dt**2],
                          [dt]])

        # Measurement matrix
        self.H = np.array([[1, 0]])

        # Process noise covariance
        self.Q = np.array([[process_noise**2 * dt**4 / 4, process_noise**2 * dt**3 / 2],
                          [process_noise**2 * dt**3 / 2, process_noise**2 * dt**2]])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise**2]])

        # Error covariance matrix
        self.P = np.array([[1000, 0],
                          [0, 1000]])

        # State vector [position, velocity]
        self.x = np.array([[0],
                          [0]])

    def predict(self):
        # Predict state
        self.x = np.dot(self.F, self.x)

        # Predict error covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement):
        # Calculate Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state
        y = measurement - np.dot(self.H, self.x)  # Innovation
        self.x = self.x + np.dot(K, y)

        # Update error covariance
        I = np.eye(len(self.x))
        self.P = np.dot((I - np.dot(K, self.H)), self.P)


# Example usage for sensor fusion
def main():
    # Create Kalman filter
    kf = KalmanFilter(dt=0.1, process_noise=0.1, measurement_noise=0.5)

    # Simulate true trajectory and measurements
    dt = 0.1
    t = np.arange(0, 10, dt)
    true_position = 10 * np.sin(0.5 * t)  # True position
    measurements = true_position + np.random.normal(0, 0.5, len(t))  # Noisy measurements

    # Store results
    estimated_positions = []

    for i, measurement in enumerate(measurements):
        # Predict step
        kf.predict()

        # Update step
        kf.update(measurement)

        # Store estimated position
        estimated_positions.append(kf.x[0, 0])

    # Plot results
    estimated_positions = np.array(estimated_positions)

    plt.figure(figsize=(12, 6))
    plt.plot(t, true_position, label='True Position', linewidth=2)
    plt.plot(t, measurements, label='Noisy Measurements', alpha=0.7)
    plt.plot(t, estimated_positions, label='Kalman Filter Estimate', linewidth=2)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('Kalman Filter for Sensor Fusion')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
```

### Particle Filtering

Particle filters are useful for non-linear, non-Gaussian systems:

```python
# Simple particle filter for robot localization
import numpy as np
import matplotlib.pyplot as plt


class ParticleFilter:
    def __init__(self, num_particles=1000, state_dim=2):
        self.num_particles = num_particles
        self.state_dim = state_dim

        # Initialize particles randomly
        self.particles = np.random.uniform(-10, 10, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input, noise_std=0.1):
        # Move particles according to motion model
        self.particles += control_input + np.random.normal(0, noise_std, self.particles.shape)

    def update(self, measurement, measurement_std=0.5):
        # Calculate likelihood of each particle given the measurement
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        likelihood = np.exp(-0.5 * (distances / measurement_std)**2)

        # Update weights
        self.weights *= likelihood
        self.weights += 1e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)  # Normalize

    def resample(self):
        # Resample particles based on weights
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        # Calculate weighted mean of particles
        return np.average(self.particles, axis=0, weights=self.weights)


# Example usage
def main():
    # Create particle filter
    pf = ParticleFilter(num_particles=1000, state_dim=2)

    # Simulate true trajectory
    dt = 0.1
    t = np.arange(0, 10, dt)
    true_trajectory = np.column_stack([2 * np.sin(0.5 * t), 2 * np.cos(0.5 * t)])

    # Store estimates
    estimates = []

    for i, true_pos in enumerate(true_trajectory):
        # Add noise to true position to simulate measurement
        measurement = true_pos + np.random.normal(0, 0.3, 2)

        # Predict and update
        control_input = np.array([0.1, 0.0]) if i > 0 else np.array([0.0, 0.0])
        pf.predict(control_input, noise_std=0.1)
        pf.update(measurement, measurement_std=0.5)

        # Resample if effective sample size is too low
        effective_samples = 1.0 / np.sum(pf.weights**2)
        if effective_samples < 0.5 * pf.num_particles:
            pf.resample()

        # Estimate position
        estimate = pf.estimate()
        estimates.append(estimate)

    # Convert to arrays
    estimates = np.array(estimates)

    # Plot results
    plt.figure(figsize=(10, 8))
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'g-', label='True Trajectory', linewidth=2)
    plt.plot(estimates[:, 0], estimates[:, 1], 'r-', label='Particle Filter Estimate', linewidth=2)
    plt.scatter(true_trajectory[::10, 0], true_trajectory[::10, 1], c='g', s=50, label='True Positions', alpha=0.7)
    plt.scatter(estimates[::10, 0], estimates[::10, 1], c='r', s=50, label='Estimates', alpha=0.7)
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Particle Filter for Robot Localization')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
```

## ROS 2 Sensor Integration

### Sensor Message Types

ROS 2 provides standardized message types for common sensors:

- `sensor_msgs/Image`: Camera images
- `sensor_msgs/LaserScan`: LIDAR scans
- `sensor_msgs/PointCloud2`: 3D point cloud data
- `sensor_msgs/Imu`: Inertial measurement unit data
- `sensor_msgs/JointState`: Joint positions, velocities, efforts

### Creating a Sensor Integration Node

```python
# Complete sensor integration example
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import threading
import time


class SensorIntegrator(Node):
    def __init__(self):
        super().__init__('sensor_integrator')

        # Initialize data storage
        self.camera_data = None
        self.lidar_data = None
        self.imu_data = None
        self.joint_data = None

        # Create subscribers for all sensor types
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Create publisher for robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create timer for sensor processing
        self.timer = self.create_timer(0.1, self.process_sensors)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Lock for thread safety
        self.data_lock = threading.Lock()

        self.get_logger().info('Sensor integrator initialized')

    def camera_callback(self, msg):
        with self.data_lock:
            self.camera_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def lidar_callback(self, msg):
        with self.data_lock:
            self.lidar_data = {
                'ranges': np.array(msg.ranges),
                'intensities': np.array(msg.intensities),
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment
            }

    def imu_callback(self, msg):
        with self.data_lock:
            self.imu_data = {
                'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            }

    def joint_callback(self, msg):
        with self.data_lock:
            self.joint_data = {
                'names': msg.name,
                'positions': np.array(msg.position),
                'velocities': np.array(msg.velocity),
                'efforts': np.array(msg.effort)
            }

    def process_sensors(self):
        with self.data_lock:
            # Process sensor data based on available information
            cmd_vel = Twist()

            # Example: Stop if obstacle detected in front (LIDAR)
            if self.lidar_data is not None:
                ranges = self.lidar_data['ranges']
                # Get front-facing ranges (forward 30 degrees)
                front_ranges = ranges[0:15]  # Approximate front ranges
                front_ranges = front_ranges[np.isfinite(front_ranges)]  # Remove invalid readings

                if len(front_ranges) > 0 and np.min(front_ranges) < 1.0:  # Obstacle within 1m
                    cmd_vel.linear.x = 0.0  # Stop
                    self.get_logger().warn('Obstacle detected! Stopping.')
                else:
                    cmd_vel.linear.x = 0.5  # Move forward slowly
            else:
                cmd_vel.linear.x = 0.0  # Stop if no LIDAR data

            # Use IMU for stability (example: adjust based on tilt)
            if self.imu_data is not None:
                # Extract roll from quaternion (simplified)
                orientation = self.imu_data['orientation']
                # Simple check for significant tilt
                if abs(orientation[1]) > 0.2:  # Significant pitch
                    cmd_vel.angular.z = -orientation[1] * 2.0  # Counteract tilt
                    self.get_logger().info('Correcting for robot tilt')

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    sensor_integrator = SensorIntegrator()

    try:
        rclpy.spin(sensor_integrator)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_integrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Hands-On Lab: Multi-Sensor Integration System

### Objective
Create a complete sensor integration system that combines camera, LIDAR, and IMU data to navigate a robot safely.

### Prerequisites
- Completed Chapter 1-3
- ROS 2 Humble with Gazebo installed
- Basic Python and OpenCV knowledge

### Steps

1. **Create a sensor fusion package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python sensor_fusion_lab --dependencies rclpy sensor_msgs cv_bridge opencv-python numpy matplotlib
   ```

2. **Create the main sensor fusion node** (`sensor_fusion_lab/sensor_fusion_lab/sensor_fusion_node.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, LaserScan, Imu
   from geometry_msgs.msg import Twist, Vector3
   from cv_bridge import CvBridge
   import numpy as np
   import threading
   import math
   from collections import deque


   class SensorFusionNode(Node):
       def __init__(self):
           super().__init__('sensor_fusion_node')

           # Initialize data storage with history for filtering
           self.camera_data = None
           self.lidar_data = None
           self.imu_data = None

           # Data history for filtering
           self.lidar_history = deque(maxlen=5)
           self.imu_history = deque(maxlen=10)

           # Create subscribers
           self.camera_sub = self.create_subscription(
               Image, '/camera/image_raw', self.camera_callback, 10)
           self.lidar_sub = self.create_subscription(
               LaserScan, '/scan', self.lidar_callback, 10)
           self.imu_sub = self.create_subscription(
               Imu, '/imu/data', self.imu_callback, 10)

           # Create publisher for velocity commands
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

           # Create publisher for fused sensor status
           self.status_pub = self.create_publisher(
               Vector3, '/sensor_fusion/status', 10)

           # Timer for processing loop
           self.timer = self.create_timer(0.05, self.process_sensors)  # 20 Hz

           # CV bridge for image processing
           self.bridge = CvBridge()

           # Robot state
           self.linear_velocity = 0.0
           self.angular_velocity = 0.0

           # Lock for thread safety
           self.data_lock = threading.Lock()

           self.get_logger().info('Sensor fusion node initialized')

       def camera_callback(self, msg):
           with self.data_lock:
               try:
                   self.camera_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
               except Exception as e:
                   self.get_logger().error(f'Error processing camera image: {e}')

       def lidar_callback(self, msg):
           with self.data_lock:
               try:
                   ranges = np.array(msg.ranges)
                   # Filter out invalid ranges
                   ranges = np.where((ranges >= msg.range_min) & (ranges <= msg.range_max), ranges, np.inf)

                   lidar_info = {
                       'ranges': ranges,
                       'angle_min': msg.angle_min,
                       'angle_max': msg.angle_max,
                       'angle_increment': msg.angle_increment
                   }

                   # Add to history for filtering
                   self.lidar_history.append(lidar_info)

                   self.lidar_data = lidar_info
               except Exception as e:
                   self.get_logger().error(f'Error processing LIDAR data: {e}')

       def imu_callback(self, msg):
           with self.data_lock:
               try:
                   imu_info = {
                       'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                       'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                       'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                   }

                   # Add to history for filtering
                   self.imu_history.append(imu_info)

                   self.imu_data = imu_info
               except Exception as e:
                   self.get_logger().error(f'Error processing IMU data: {e}')

       def process_sensors(self):
           with self.data_lock:
               # Initialize command
               cmd_vel = Twist()

               # Process LIDAR data for obstacle avoidance
               if self.lidar_data is not None:
                   cmd_vel.linear.x, cmd_vel.angular.z = self.process_lidar_navigation()

               # Process IMU data for stability
               if self.imu_data is not None:
                   cmd_vel = self.process_imu_stability(cmd_vel)

               # Process camera data for visual features (simplified)
               if self.camera_data is not None:
                   visual_cmd = self.process_camera_navigation()
                   # Combine with other commands (simple weighted average)
                   cmd_vel.linear.x = 0.7 * cmd_vel.linear.x + 0.3 * visual_cmd.linear.x
                   cmd_vel.angular.z = 0.7 * cmd_vel.angular.z + 0.3 * visual_cmd.angular.z

               # Apply velocity limits
               cmd_vel.linear.x = max(-1.0, min(1.0, cmd_vel.linear.x))
               cmd_vel.angular.z = max(-1.0, min(1.0, cmd_vel.angular.z))

               # Publish command
               self.cmd_vel_pub.publish(cmd_vel)

               # Publish status
               status = Vector3()
               status.x = cmd_vel.linear.x
               status.y = cmd_vel.angular.z
               status.z = 1.0 if self.lidar_data is not None else 0.0
               self.status_pub.publish(status)

       def process_lidar_navigation(self):
           """Process LIDAR data for obstacle avoidance and navigation"""
           ranges = self.lidar_data['ranges']
           angle_min = self.lidar_data['angle_min']
           angle_increment = self.lidar_data['angle_increment']

           # Define sectors: front, front-left, front-right, left, right
           total_angles = len(ranges)
           front_idx = total_angles // 2
           sector_size = total_angles // 8  # 45-degree sectors

           front_range = np.min(ranges[front_idx - sector_size//2 : front_idx + sector_size//2])
           front_left_range = np.min(ranges[front_idx - sector_size : front_idx - sector_size//2])
           front_right_range = np.min(ranges[front_idx + sector_size//2 : front_idx + sector_size])
           left_range = np.min(ranges[0 : sector_size])
           right_range = np.min(ranges[-sector_size :])

           # Obstacle avoidance logic
           min_distance = 0.8  # meters
           target_speed = 0.5  # m/s

           linear_vel = target_speed
           angular_vel = 0.0

           # If obstacle is very close, stop and turn
           if front_range < min_distance * 0.7:
               linear_vel = 0.0
               # Turn away from the closest obstacle
               if front_left_range < front_right_range:
                   angular_vel = -0.8  # Turn right
               else:
                   angular_vel = 0.8   # Turn left
           # If obstacle is close in front, slow down and turn
           elif front_range < min_distance:
               linear_vel = target_speed * 0.3
               if front_left_range < front_right_range:
                   angular_vel = -0.4
               else:
                   angular_vel = 0.4
           # If obstacle is to the side, gently turn away
           elif left_range < min_distance * 1.2:
               angular_vel = 0.3
           elif right_range < min_distance * 1.2:
               angular_vel = -0.3

           return linear_vel, angular_vel

       def process_imu_stability(self, cmd_vel):
           """Process IMU data for robot stability"""
           # Extract orientation from quaternion (simplified approach)
           orientation = self.imu_data['orientation']

           # Convert quaternion to roll/pitch (simplified)
           # This is a basic approximation - in practice, use proper quaternion math
           sinr_cosp = 2 * (orientation[3] * orientation[0] + orientation[1] * orientation[2])
           cosr_cosp = 1 - 2 * (orientation[0]**2 + orientation[1]**2)
           roll = math.atan2(sinr_cosp, cosr_cosp)

           sinp = 2 * (orientation[3] * orientation[1] - orientation[2] * orientation[0])
           # Check for gimbal lock
           if abs(sinp) >= 1:
               pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
           else:
               pitch = math.asin(sinp)

           # If robot is tilted too much, reduce speed
           tilt_threshold = 0.3  # radians
           if abs(pitch) > tilt_threshold or abs(roll) > tilt_threshold:
               cmd_vel.linear.x *= 0.3  # Reduce speed significantly
               cmd_vel.angular.z *= 0.7  # Reduce turning
               self.get_logger().warn('Robot tilt detected, reducing speed')

           return cmd_vel

       def process_camera_navigation(self):
           """Process camera data for visual navigation (simplified)"""
           # This is a simplified example - in practice, this would involve
           # complex computer vision algorithms

           import cv2

           # Convert to grayscale
           gray = cv2.cvtColor(self.camera_data, cv2.COLOR_BGR2GRAY)

           # Simple edge detection to find obstacles or features
           edges = cv2.Canny(gray, 50, 150)

           # Calculate the center of mass of edges to estimate where to go
           # (simplified - in practice, use more sophisticated methods)
           height, width = edges.shape
           x_coords, y_coords = np.where(edges > 0)

           cmd_vel = Twist()

           if len(x_coords) > 0:
               # Calculate centroid of detected edges
               centroid_x = np.mean(y_coords)  # x in image coordinates
               center_x = width / 2

               # If edges are mostly on the left, turn right; if on right, turn left
               if centroid_x < center_x - 20:  # Left side has more edges
                   cmd_vel.angular.z = 0.3
               elif centroid_x > center_x + 20:  # Right side has more edges
                   cmd_vel.angular.z = -0.3

               # Reduce speed if many edges detected (potential obstacle)
               if len(x_coords) > height * width * 0.1:  # If more than 10% of pixels are edges
                   cmd_vel.linear.x = 0.2
               else:
                   cmd_vel.linear.x = 0.5
           else:
               # No significant features detected, go forward
               cmd_vel.linear.x = 0.5

           return cmd_vel


   def main(args=None):
       rclpy.init(args=args)
       sensor_fusion_node = SensorFusionNode()

       try:
           rclpy.spin(sensor_fusion_node)
       except KeyboardInterrupt:
           pass
       finally:
           sensor_fusion_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`sensor_fusion_lab/launch/sensor_fusion.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='true',
           description='Use simulation (Gazebo) clock if true'
       )

       # Include Gazebo launch (if needed)
       # gazebo = IncludeLaunchDescription(
       #     PythonLaunchDescriptionSource([
       #         get_package_share_directory('gazebo_ros'),
       #         '/launch/gzserver.launch.py'
       #     ])
       # )

       # Sensor fusion node
       sensor_fusion_node = Node(
           package='sensor_fusion_lab',
           executable='sensor_fusion_node',
           name='sensor_fusion_node',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
           output='screen'
       )

       return LaunchDescription([
           use_sim_time,
           # gazebo,
           sensor_fusion_node
       ])
   ```

4. **Update setup.py** to include the executable:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'sensor_fusion_lab'

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
       description='Sensor fusion lab for robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'sensor_fusion_node = sensor_fusion_lab.sensor_fusion_node:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select sensor_fusion_lab
   source install/setup.bash
   ```

6. **Run the sensor fusion system**:
   ```bash
   ros2 launch sensor_fusion_lab sensor_fusion.launch.py
   ```

### Expected Results
- The robot should navigate using a combination of sensor inputs
- LIDAR data should be used for obstacle avoidance
- IMU data should be used for stability
- Camera data should provide visual guidance
- The system should demonstrate basic sensor fusion principles

### Troubleshooting Tips
- Ensure all sensor topics are being published
- Check that the robot model has all required sensors
- Verify that the sensor fusion node has the correct topic names
- Monitor the robot's behavior to ensure safe operation

## Summary

In this chapter, we've explored the fundamental concepts of sensor integration and data processing for robotics. We covered various types of sensors commonly used in robotics, techniques for processing sensor data, and methods for fusing information from multiple sensors.

The hands-on lab provided practical experience in creating a complete sensor fusion system that combines camera, LIDAR, and IMU data to enable safe navigation. This foundation is essential for more advanced perception systems that we'll explore in the next chapters, including computer vision for robotics and 3D perception.