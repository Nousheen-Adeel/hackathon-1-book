---
sidebar_position: 3
title: "Robot Simulation (Gazebo & Unity)"
---

# Robot Simulation (Gazebo & Unity)

## Weekly Plan
- Day 1-2: Understanding Gazebo physics simulation and world creation
- Day 3-4: Creating robot models with URDF/SDF and sensors
- Day 5-7: Advanced simulation features and Unity integration

## Learning Objectives
By the end of this chapter, you will:
- Create and configure robot models in URDF/SDF formats
- Set up physics-based simulations with realistic dynamics
- Integrate sensors into simulated robots
- Use Unity for advanced visualization and simulation
- Connect simulated robots to ROS 2 systems

## Gazebo Simulation Overview

Gazebo is a physics-based simulation environment that provides realistic dynamics, sensor simulation, and rendering capabilities. It's essential for testing robotic systems before deployment on real hardware.

### Key Features
- Physics engine with collision detection
- Sensor simulation (cameras, LIDAR, IMU, etc.)
- Realistic rendering and visualization
- ROS 2 integration through gazebo_ros_pkgs
- Plugin architecture for custom behaviors

## Creating Robot Models with URDF

URDF (Unified Robot Description Format) defines the physical and kinematic properties of robots:

```xml
<!-- my_robot.urdf -->
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Wheel links -->
  <link name="wheel_left">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="0.15 -0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Differential drive controller -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>
</robot>
```

## Creating World Files

World files define the environment for simulation:

```xml
<!-- simple_room.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Walls -->
    <model name="wall_1">
      <pose>0 5 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture -->
    <model name="table">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="table_base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Adding Sensors to Robots

### Camera Sensor
```xml
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>image_raw</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### LIDAR Sensor
```xml
<gazebo reference="lidar_link">
  <sensor type="ray" name="lidar_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
      <topic_name>scan</topic_name>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Gazebo-ROS 2 Integration

### Launching Simulation with ROS 2
```python
# launch_simulation.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'worlds',
                'simple_room.world'
            ])
        }.items()
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0',
            '-y', '0',
            '-z', '0.2'
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'my_robot.urdf'
            ])
        }]
    )

    return LaunchDescription([
        gazebo,
        spawn_entity,
        robot_state_publisher
    ])
```

## Physics Configuration

### Physics Parameters
```xml
<!-- In world file -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Unity Integration for Advanced Visualization

Unity can be used for more advanced visualization and simulation:

### Setting up Unity Robotics
```csharp
// UnityRobotController.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotTopic = "unity_robot_pose";

    // Start is called before the first frame update
    void Start()
    {
        ros = ROSConnection.instance;
        InvokeRepeating("SendRobotPose", 0.1f, 0.1f);
    }

    void SendRobotPose()
    {
        // Create and publish robot pose message
        PoseMsg pose = new PoseMsg();
        pose.position.x = transform.position.x;
        pose.position.y = transform.position.y;
        pose.position.z = transform.position.z;

        pose.orientation.x = transform.rotation.x;
        pose.orientation.y = transform.rotation.y;
        pose.orientation.z = transform.rotation.z;
        pose.orientation.w = transform.rotation.w;

        ros.Publish(robotTopic, pose);
    }

    void OnMessageReceived(PoseMsg pose)
    {
        // Update robot position based on ROS message
        transform.position = new Vector3(
            (float)pose.position.x,
            (float)pose.position.y,
            (float)pose.position.z
        );

        transform.rotation = new Quaternion(
            (float)pose.orientation.x,
            (float)pose.orientation.y,
            (float)pose.orientation.z,
            (float)pose.orientation.w
        );
    }
}
```

## Simulation Best Practices

### Performance Optimization
```python
# Efficient simulation node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class SimulationOptimizedNode(Node):
    def __init__(self):
        super().__init__('simulation_optimized')

        # Use appropriate QoS for simulation
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Throttle processing
        self.processing_rate = 10  # Hz
        self.timer = self.create_timer(
            1.0 / self.processing_rate,
            self.process_scan
        )

        self.latest_scan = None
        self.obstacle_detected = False

    def scan_callback(self, msg):
        self.latest_scan = msg

    def process_scan(self):
        if self.latest_scan is None:
            return

        # Simple obstacle detection
        min_range = min(self.latest_scan.ranges)
        self.obstacle_detected = min_range < 1.0  # 1 meter threshold

        # Send command based on obstacle detection
        cmd = Twist()
        if self.obstacle_detected:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn
        else:
            cmd.linear.x = 0.5  # Forward
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SimulationOptimizedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Weekly Exercises

### Exercise 1: Robot Model Creation
1. Create a URDF model of a simple differential drive robot
2. Add visual and collision properties
3. Include a differential drive controller plugin
4. Test the model in Gazebo

### Exercise 2: Sensor Integration
1. Add a camera sensor to your robot model
2. Add a LIDAR sensor to your robot model
3. Verify that sensor data is published to ROS topics
4. Visualize the sensor data in RViz

### Exercise 3: World Creation
1. Create a custom world file with obstacles
2. Add furniture and navigation challenges
3. Test your robot's navigation in the custom world
4. Adjust physics parameters for realistic behavior

### Mini-Project: Autonomous Navigation Simulation
Create a complete simulation environment with:
- Custom robot model with sensors
- Complex world with obstacles
- Navigation stack integration
- Autonomous navigation demonstration

```python
# navigation_simulation.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math

class NavigationSimulator(Node):
    def __init__(self):
        super().__init__('navigation_simulator')

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )

        # Navigation parameters
        self.target_x = 5.0
        self.target_y = 5.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Control loop
        self.timer = self.create_timer(0.1, self.navigation_loop)

        self.get_logger().info('Navigation simulator started')

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def scan_callback(self, msg):
        # Store latest scan for navigation
        self.latest_scan = msg

    def navigation_loop(self):
        if not hasattr(self, 'latest_scan'):
            return

        # Calculate distance to target
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate target angle
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.current_yaw

        # Normalize angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Simple navigation controller
        cmd = Twist()

        # Check for obstacles
        min_range = min(self.latest_scan.ranges) if self.latest_scan.ranges else float('inf')

        if min_range < 0.5:  # Obstacle too close
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn to avoid
        elif abs(angle_diff) > 0.1:  # Need to turn
            cmd.angular.z = max(-0.5, min(0.5, angle_diff * 2.0))
        else:  # Move forward if not too close to target
            cmd.linear.x = 0.5 if distance > 0.5 else 0.0

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = NavigationSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary
This chapter covered robot simulation using Gazebo and Unity, including URDF/SDF model creation, sensor integration, and physics configuration. You've learned how to create realistic simulation environments for testing robotic systems. The next chapter will explore the NVIDIA Isaac platform for AI-powered robotics.