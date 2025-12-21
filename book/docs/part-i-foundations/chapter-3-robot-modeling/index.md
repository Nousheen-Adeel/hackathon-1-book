---
title: Chapter 3 - Robot Modeling and Simulation Fundamentals
sidebar_position: 3
---

# Chapter 3: Robot Modeling and Simulation Fundamentals

## Learning Goals

- Understand URDF and SDF robot description formats
- Learn kinematic and dynamic modeling concepts
- Master simulation environment setup
- Create URDF model of a simple humanoid robot
- Simulate robot in Gazebo environment
- Visualize robot in RViz

## Introduction to Robot Modeling

Robot modeling is the process of creating digital representations of physical robots that can be used for simulation, visualization, and control development. In robotics, we use standardized formats to describe robot geometry, kinematics, dynamics, and sensors. The two primary formats in ROS are URDF (Unified Robot Description Format) for ROS-based systems and SDF (Simulation Description Format) for Gazebo simulation.

### Why Robot Modeling Matters

Robot modeling is crucial for several reasons:

1. **Simulation**: Test algorithms without physical hardware
2. **Visualization**: Understand robot kinematics and motion
3. **Control Development**: Develop and test controllers in a safe environment
4. **Collision Detection**: Prevent self-collision and environment collision
5. **Sensor Simulation**: Test perception algorithms with realistic sensor data

## Unified Robot Description Format (URDF)

URDF is an XML-based format that describes robot models in ROS. It defines the robot's physical properties including links (rigid bodies), joints (connections between links), and other elements like sensors and actuators.

### URDF Structure

A URDF file contains several key elements:

- **Links**: Rigid bodies that make up the robot
- **Joints**: Connections between links with specific degrees of freedom
- **Materials**: Visual properties like color and texture
- **Gazebo plugins**: Simulation-specific extensions

### Basic URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Arm link -->
  <link name="arm_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.5"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joint connecting base and arm -->
  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

### URDF Links

Links represent rigid bodies in the robot. Each link can have:

- **Visual**: How the link appears in visualization tools
- **Collision**: How the link interacts in collision detection
- **Inertial**: Physical properties for dynamics simulation

```xml
<link name="example_link">
  <!-- Visual properties -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Can be box, cylinder, sphere, or mesh -->
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="green">
      <color rgba="0 0.8 0 1"/>
    </material>
  </visual>

  <!-- Collision properties -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>

  <!-- Inertial properties -->
  <inertial>
    <mass value="0.5"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>
```

### URDF Joints

Joints define how links connect and move relative to each other. Common joint types include:

- **Revolute**: Rotational joint with limited range
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear sliding joint
- **Fixed**: No movement (welded connection)
- **Floating**: 6 DOF (used for base of mobile robots)

```xml
<!-- Revolute joint example -->
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>

<!-- Fixed joint example -->
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>
```

## Xacro: XML Macros for URDF

Xacro is a macro language that extends URDF, allowing you to create more maintainable and reusable robot descriptions through variables, properties, and includes.

### Basic Xacro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_radius" value="0.2" />
  <xacro:property name="base_length" value="0.6" />

  <!-- Define a macro for a wheel -->
  <xacro:macro name="wheel" params="prefix parent xyz">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="0.1" length="0.05"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="0.1" length="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Use the base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="${base_length}" radius="${base_radius}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${base_length}" radius="${base_radius}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="2.0"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.15 0.15 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.15 -0.15 0"/>
  <xacro:wheel prefix="rear_left" parent="base_link" xyz="-0.15 0.15 0"/>
  <xacro:wheel prefix="rear_right" parent="base_link" xyz="-0.15 -0.15 0"/>

</robot>
```

## Simulation with Gazebo

Gazebo is a 3D simulation environment that provides high-fidelity physics simulation, realistic rendering, and convenient programmatic interfaces. It's commonly used with ROS for robot simulation.

### Gazebo Integration with URDF

To use your URDF model in Gazebo, you need to add Gazebo-specific extensions:

```xml
<!-- Add Gazebo-specific properties -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <turnGravityOff>false</turnGravityOff>
</gazebo>

<!-- Add transmission for joint control -->
<transmission name="arm_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="arm_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="arm_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>

<!-- Add gazebo plugin for ROS control -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/simple_robot</robotNamespace>
  </plugin>
</gazebo>
```

### Launching Gazebo with Your Robot

Create a launch file to spawn your robot in Gazebo:

```python
# launch/robot_spawn.launch.py
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='simple_robot',
        description='Model name for the robot'
    )

    # Get URDF file path
    robot_description_path = PathJoinSubstitution([
        get_package_share_directory('your_robot_package'),
        'urdf',
        LaunchConfiguration('model') + '.urdf.xacro'
    ])

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_path}]
    )

    # Gazebo server
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/gzserver.launch.py'
        ])
    )

    # Gazebo client (GUI)
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/gzclient.launch.py'
        ])
    )

    # Spawn entity in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', LaunchConfiguration('model')
        ],
        output='screen'
    )

    return LaunchDescription([
        model_arg,
        robot_state_publisher,
        gzserver,
        gzclient,
        spawn_entity
    ])
```

## RViz Visualization

RViz is ROS's 3D visualization tool that allows you to visualize robot models, sensor data, paths, and other ROS messages in a 3D environment.

### Basic RViz Configuration

Create an RViz configuration file to properly visualize your robot:

```yaml
# config/robot_view.rviz
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
      Splitter Ratio: 0.5
    Tree Height: 617
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      Robot Description: robot_description
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Name: Current View
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Width: 1200
```

## Creating a Simple Humanoid Robot Model

Let's create a simplified humanoid robot model using URDF and Xacro:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_humanoid">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="body_mass" value="10.0" />
  <xacro:property name="limb_mass" value="2.0" />
  <xacro:property name="head_mass" value="1.0" />

  <!-- Materials -->
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>

  <!-- Base Footprint (for navigation) -->
  <link name="base_footprint">
    <visual>
      <geometry>
        <sphere radius="0.01"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.01"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.0001"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Base Link (torso) -->
  <joint name="base_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 0.7" rpy="0 0 0"/>
  </joint>

  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.4" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.8"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.4" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.8"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${body_mass}"/>
      <origin xyz="0 0 0.4" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.85" rpy="0 0 0"/>
  </joint>

  <link name="head_link">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${head_mass}"/>
      <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.004"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Arm (mirror of left) -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.15 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.07 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="0" effort="100" velocity="1"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right Leg (mirror of left) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.07 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
  </joint>

  <link name="right_upper_leg">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="0" effort="100" velocity="1"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/simple_humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Joint state publisher for visualization -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <robotNamespace>/simple_humanoid</robotNamespace>
      <jointName>left_shoulder_joint</jointName>
      <jointName>left_elbow_joint</jointName>
      <jointName>right_shoulder_joint</jointName>
      <jointName>right_elbow_joint</jointName>
      <jointName>left_hip_joint</jointName>
      <jointName>left_knee_joint</jointName>
      <jointName>right_hip_joint</jointName>
      <jointName>right_knee_joint</jointName>
    </plugin>
  </gazebo>

</robot>
```

## Working with Robot State Publisher

The robot_state_publisher node is crucial for visualizing your robot in RViz. It reads the robot description parameter and joint states to publish the TF tree.

```python
# Example of setting up robot state publisher in a launch file
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the URDF file path
    urdf_file = os.path.join(
        get_package_share_directory('your_robot_package'),
        'urdf',
        'simple_humanoid.urdf.xacro'
    )

    # Read the URDF file
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc,
            'publish_frequency': 50.0
        }]
    )

    # Joint State Publisher (for visualization)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{
            'source_list': ['joint_states'],
            'rate': 50.0
        }]
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher
    ])
```

## Hands-On Lab: Create and Visualize Your Robot Model

### Objective
Create a complete robot model using URDF/Xacro, visualize it in RViz, and simulate it in Gazebo.

### Prerequisites
- Completed Chapter 1 and 2
- ROS 2 Humble with Gazebo installed

### Steps

1. **Create a robot description package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_cmake robot_description --dependencies urdf xacro
   ```

2. **Create the URDF directory structure**:
   ```bash
   mkdir -p robot_description/urdf
   mkdir -p robot_description/meshes
   mkdir -p robot_description/config
   mkdir -p robot_description/launch
   ```

3. **Create the robot model file** (`robot_description/urdf/simple_humanoid.urdf.xacro`):
   Copy the humanoid robot URDF code from the previous section into this file.

4. **Create a launch file for visualization** (`robot_description/launch/robot_visualize.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory


   def generate_launch_description():
       # Declare launch arguments
       model_arg = DeclareLaunchArgument(
           'model',
           default_value='simple_humanoid.urdf.xacro',
           description='Robot description file'
       )

       # Get URDF file path
       urdf_file = PathJoinSubstitution([
           get_package_share_directory('robot_description'),
           'urdf',
           LaunchConfiguration('model')
       ])

       # Robot State Publisher node
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           parameters=[{
               'robot_description': urdf_file,
               'publish_frequency': 50.0
           }]
       )

       # Joint State Publisher (GUI for testing joint movement)
       joint_state_publisher_gui = Node(
           package='joint_state_publisher_gui',
           executable='joint_state_publisher_gui',
           name='joint_state_publisher_gui'
       )

       # RViz2 node
       rviz = Node(
           package='rviz2',
           executable='rviz2',
           name='rviz2',
           arguments=['-d', PathJoinSubstitution([
               get_package_share_directory('robot_description'),
               'config',
               'robot_view.rviz'
           ])]
       )

       return LaunchDescription([
           model_arg,
           robot_state_publisher,
           joint_state_publisher_gui,
           rviz
       ])
   ```

5. **Create the RViz configuration file** (`robot_description/config/robot_view.rviz`):
   Copy the RViz configuration from the previous section into this file.

6. **Create a launch file for Gazebo simulation** (`robot_description/launch/robot_gazebo.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory


   def generate_launch_description():
       # Declare launch arguments
       model_arg = DeclareLaunchArgument(
           'model',
           default_value='simple_humanoid.urdf.xacro',
           description='Robot description file'
       )

       # Get URDF file path
       urdf_file = PathJoinSubstitution([
           get_package_share_directory('robot_description'),
           'urdf',
           LaunchConfiguration('model')
       ])

       # Robot State Publisher node
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           parameters=[{
               'robot_description': urdf_file,
               'publish_frequency': 50.0
           }]
       )

       # Gazebo server
       gzserver = IncludeLaunchDescription(
           PythonLaunchDescriptionSource([
               get_package_share_directory('gazebo_ros'),
               '/launch/gzserver.launch.py'
           ])
       )

       # Gazebo client (GUI)
       gzclient = IncludeLaunchDescription(
           PythonLaunchDescriptionSource([
               get_package_share_directory('gazebo_ros'),
               '/launch/gzclient.launch.py'
           ])
       )

       # Spawn entity in Gazebo
       spawn_entity = Node(
           package='gazebo_ros',
           executable='spawn_entity.py',
           arguments=[
               '-file', urdf_file,
               '-entity', 'simple_humanoid'
           ],
           output='screen'
       )

       return LaunchDescription([
           model_arg,
           robot_state_publisher,
           gzserver,
           gzclient,
           spawn_entity
       ])
   ```

7. **Update the package.xml file** (`robot_description/package.xml`):
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>robot_description</name>
     <version>0.0.0</version>
     <description>Robot description package for simple humanoid</description>
     <maintainer email="your.email@example.com">Your Name</maintainer>
     <license>Apache License 2.0</license>

     <buildtool_depend>ament_cmake</buildtool_depend>

     <depend>urdf</depend>
     <depend>xacro</depend>

     <test_depend>ament_lint_auto</test_depend>
     <test_depend>ament_lint_common</test_depend>

     <export>
       <build_type>ament_cmake</build_type>
     </export>
   </package>
   ```

8. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select robot_description
   source install/setup.bash
   ```

9. **Visualize the robot in RViz**:
   ```bash
   ros2 launch robot_description robot_visualize.launch.py
   ```

10. **Simulate the robot in Gazebo**:
    ```bash
    ros2 launch robot_description robot_gazebo.launch.py
    ```

### Expected Results
- The robot model should appear in RViz with all links properly connected
- Joint state publisher GUI should allow you to move the joints and see the robot move
- The robot should spawn correctly in Gazebo simulation
- TF frames should be published correctly for all robot parts

### Troubleshooting Tips
- Ensure the URDF file is valid XML and properly formatted
- Check that joint limits are reasonable and not causing issues
- Verify that the robot model doesn't have self-collisions in its default pose
- Make sure all required packages (urdf, xacro, robot_state_publisher) are installed

## Summary

In this chapter, we've covered the fundamentals of robot modeling using URDF and Xacro, including how to create complex robot models with multiple links and joints. We've also explored simulation with Gazebo and visualization with RViz, which are essential tools for robotics development.

The hands-on lab provided practical experience in creating a complete humanoid robot model and visualizing it in both RViz and Gazebo. These skills are fundamental for any robotics application, as they allow you to test and develop algorithms in a safe, simulated environment before deploying to real hardware.

In the next chapters, we'll build upon these foundations to explore perception systems, control algorithms, and intelligent behaviors for humanoid robots.