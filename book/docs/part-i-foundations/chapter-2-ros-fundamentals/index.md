---
title: Chapter 2 - Robot Operating System (ROS 2) Fundamentals
sidebar_position: 2
---

# Chapter 2: Robot Operating System (ROS 2) Fundamentals

## Learning Goals

- Master ROS 2 architecture and communication patterns
- Understand nodes, topics, services, and actions
- Learn about parameter management and launch files
- Build a multi-node system for robot control
- Implement custom message types and services
- Create launch files for complex robot systems

## Introduction to ROS 2

The Robot Operating System 2 (ROS 2) is not an actual operating system, but rather a flexible framework for writing robot software. It provides services designed for a heterogeneous computer cluster, including hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

ROS 2 is the second generation of the Robot Operating System, designed to address the limitations of ROS 1 and to provide features needed for production robotics applications, including improved security, real-time support, and better cross-platform compatibility.

### Evolution from ROS 1 to ROS 2

ROS 2 was developed to address several key limitations of ROS 1:

- **Real-time support**: ROS 2 provides better support for real-time systems
- **Multi-robot systems**: Improved support for multiple robots and distributed systems
- **Production deployment**: Better tools and practices for deploying ROS in production
- **Cross-platform support**: Expanded platform support including Windows and macOS
- **Security**: Built-in security features for protected communication
- **DDS Integration**: Uses Data Distribution Service (DDS) as the underlying communication middleware

## ROS 2 Architecture

### Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of a ROS 2 system. Each node is designed to perform specific computations and can communicate with other nodes through various mechanisms.

In ROS 2, nodes are more robust than in ROS 1. They can be created and destroyed more easily, and they provide better introspection capabilities.

```python
# Example of a basic ROS 2 node
import rclpy
from rclpy.node import Node


class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from minimal node!')


def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()

    # Keep the node running
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Topics and Message Passing

Topics are named buses over which nodes exchange messages. They implement a publish/subscribe communication pattern where publishers send messages to a topic and subscribers receive messages from a topic.

The communication is asynchronous - publishers don't wait for subscribers and vice versa. Multiple publishers can publish to the same topic, and multiple subscribers can subscribe to the same topic.

```python
# Publisher example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

```python
# Subscriber example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Services

Services implement a request/reply communication pattern. A service client sends a request to a service server, which processes the request and sends back a response. This is synchronous communication - the client waits for the response.

Services are defined by `.srv` files that specify the request and response message types.

```python
# Service definition example (in srv/AddTwoInts.srv):
# int64 a
# int64 b
# ---
# int64 sum
```

```python
# Service server example
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response


def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

```python
# Service client example
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Actions

Actions are a more advanced communication pattern that allows for long-running tasks with feedback and goal management. They're ideal for tasks like navigation, where you want to track progress and potentially cancel the operation.

Actions consist of three message types:
- Goal: What the action should do
- Result: What happened when the action completed
- Feedback: Periodic updates on progress

```python
# Action example for navigation
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose


class NavigateToPoseActionServer(Node):
    def __init__(self):
        super().__init__('navigate_to_pose_action_server')
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        # Simulate navigation progress
        feedback_msg = NavigateToPose.Feedback()
        feedback_msg.current_pose = goal_handle.request.pose

        # Simulate navigation
        for i in range(0, 101, 10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return NavigateToPose.Result()

            feedback_msg.feedback = f'Navigating: {i}% complete'
            goal_handle.publish_feedback(feedback_msg)

            time.sleep(0.5)  # Simulate work

        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.result = True
        self.get_logger().info('Goal succeeded')
        return result


def main(args=None):
    rclpy.init(args=args)
    action_server = NavigateToPoseActionServer()
    rclpy.spin(action_server)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Parameters

Parameters in ROS 2 are named values that can be set at runtime and changed dynamically. They provide a way to configure nodes without recompiling. Parameters can be set through launch files, command line, or programmatically.

```python
# Parameter usage example
import rclpy
from rclpy.node import Node


class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_enabled', True)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_enabled = self.get_parameter('safety_enabled').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Safety enabled: {self.safety_enabled}')

        # Set up parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.PARAMETER_DOUBLE:
                if param.value > 5.0:
                    return SetParametersResult(successful=False, reason='Max velocity too high')
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch Files

Launch files in ROS 2 allow you to start multiple nodes with a single command. They provide a way to configure and start complex systems with many interconnected nodes. Launch files are written in Python and offer powerful features like conditional launching, parameter setting, and node remapping.

```python
# Example launch file (launch/robot_system.launch.py)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare launch argument
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    # Create nodes
    minimal_publisher = Node(
        package='demo_nodes_py',
        executable='talker',
        name='publisher_node',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('chatter', 'robot_messages')]
    )

    minimal_subscriber = Node(
        package='demo_nodes_py',
        executable='listener',
        name='subscriber_node',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)

    # Add nodes
    ld.add_action(minimal_publisher)
    ld.add_action(minimal_subscriber)

    return ld
```

## Creating Custom Message Types

ROS 2 allows you to define custom message types for your specific applications. Messages are defined using the Interface Definition Language (IDL) and are stored in `.msg` files.

### Defining a Custom Message

Create a file named `RobotStatus.msg` in your package's `msg/` directory:

```
# RobotStatus.msg
string robot_name
float64 battery_level
bool is_charging
int32[] joint_positions
geometry_msgs/Pose current_pose
```

### Using Custom Messages

```python
# Publisher using custom message
import rclpy
from rclpy.node import Node
from your_package_name.msg import RobotStatus  # Import your custom message
from geometry_msgs.msg import Pose


class RobotStatusPublisher(Node):
    def __init__(self):
        super().__init__('robot_status_publisher')
        self.publisher_ = self.create_publisher(RobotStatus, 'robot_status', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = RobotStatus()
        msg.robot_name = "MyRobot"
        msg.battery_level = 85.5
        msg.is_charging = False
        msg.joint_positions = [0, 45, 90, -45, 0, 30]  # Example joint angles

        # Set pose
        msg.current_pose.position.x = 1.0
        msg.current_pose.position.y = 2.0
        msg.current_pose.position.z = 0.0
        msg.current_pose.orientation.w = 1.0

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published robot status for {msg.robot_name}')


def main(args=None):
    rclpy.init(args=args)
    robot_status_publisher = RobotStatusPublisher()
    rclpy.spin(robot_status_publisher)
    robot_status_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Advanced Communication Patterns

### Quality of Service (QoS) Settings

QoS settings allow you to control the reliability and durability of message transmission:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a QoS profile for reliable communication
reliable_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Create a QoS profile for best-effort communication
best_effort_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# Use in publisher
publisher = self.create_publisher(String, 'topic', reliable_qos)
```

### Lifecycle Nodes

Lifecycle nodes provide a way to manage the state of nodes through a well-defined state machine:

```python
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState


class LifecycleDemoNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_demo_node')
        self.get_logger().info('Constructor called.')

    def on_configure(self, state: LifecycleState):
        self.get_logger().info('on_configure() is called.')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        self.get_logger().info('on_activate() is called.')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        self.get_logger().info('on_deactivate() is called.')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        self.get_logger().info('on_cleanup() is called.')
        return TransitionCallbackReturn.SUCCESS
```

## Hands-On Lab: Multi-Node Robot Control System

### Objective
Create a multi-node system that simulates robot control with status monitoring and command processing.

### Prerequisites
- Completed Chapter 1 setup
- ROS 2 Humble installed

### Steps

1. **Create a new package** for the lab:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python robot_control_lab --dependencies rclpy std_msgs geometry_msgs
   ```

2. **Create the robot controller node** (`robot_control_lab/robot_control_lab/robot_controller.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist
   import time


   class RobotController(Node):
       def __init__(self):
           super().__init__('robot_controller')

           # Create publisher for velocity commands
           self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

           # Create subscriber for movement commands
           self.cmd_sub = self.create_subscription(
               String,
               'movement_commands',
               self.command_callback,
               10
           )

           # Create timer for status updates
           self.status_timer = self.create_timer(1.0, self.publish_status)

           self.get_logger().info('Robot Controller initialized')

       def command_callback(self, msg):
           command = msg.data.lower()
           twist = Twist()

           if command == 'forward':
               twist.linear.x = 1.0
           elif command == 'backward':
               twist.linear.x = -1.0
           elif command == 'left':
               twist.angular.z = 1.0
           elif command == 'right':
               twist.angular.z = -1.0
           elif command == 'stop':
               twist.linear.x = 0.0
               twist.angular.z = 0.0
           else:
               self.get_logger().warn(f'Unknown command: {command}')
               return

           self.cmd_vel_pub.publish(twist)
           self.get_logger().info(f'Executing command: {command}')

       def publish_status(self):
           status_msg = String()
           status_msg.data = f'Robot status: OK - {time.time()}'
           status_pub = self.create_publisher(String, 'robot_status', 10)
           status_pub.publish(status_msg)


   def main(args=None):
       rclpy.init(args=args)
       controller = RobotController()
       rclpy.spin(controller)
       controller.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create the command interface node** (`robot_control_lab/robot_control_lab/command_interface.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   import sys


   class CommandInterface(Node):
       def __init__(self):
           super().__init__('command_interface')
           self.publisher = self.create_publisher(String, 'movement_commands', 10)
           self.get_logger().info('Command Interface ready. Send commands: forward, backward, left, right, stop')

       def send_command(self, command):
           msg = String()
           msg.data = command
           self.publisher.publish(msg)
           self.get_logger().info(f'Sent command: {command}')


   def main(args=None):
       rclpy.init(args=args)
       interface = CommandInterface()

       if len(sys.argv) > 1:
           command = sys.argv[1]
           interface.send_command(command)
       else:
           print("Usage: ros2 run robot_control_lab command_interface <command>")
           print("Commands: forward, backward, left, right, stop")

       # Keep node alive briefly to send message
       interface.send_command('stop')  # Ensure robot stops
       interface.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

4. **Update setup.py** to include executables:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'robot_control_lab'

   setup(
       name=package_name,
       version='0.0.0',
       packages=find_packages(exclude=['test']),
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Robot Control Lab for ROS 2',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'robot_controller = robot_control_lab.robot_controller:main',
               'command_interface = robot_control_lab.command_interface:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select robot_control_lab
   source install/setup.bash
   ```

6. **Run the multi-node system**:
   ```bash
   # Terminal 1: Start the robot controller
   ros2 run robot_control_lab robot_controller

   # Terminal 2: Send commands
   ros2 run robot_control_lab command_interface forward
   ros2 run robot_control_lab command_interface left
   ros2 run robot_control_lab command_interface stop
   ```

### Expected Results
- The robot controller node should respond to movement commands
- Velocity commands should be published to the `/cmd_vel` topic
- Status messages should be published periodically
- The system should demonstrate proper node communication

### Troubleshooting Tips
- Ensure all packages are built and sourced
- Check topic names match between publisher and subscriber
- Verify node names don't conflict
- Use `ros2 topic list` and `ros2 node list` to verify connections

## Summary

In this chapter, we've explored the fundamental concepts of ROS 2, including nodes, topics, services, actions, parameters, and launch files. You've learned how to create multi-node systems and implement custom message types. The hands-on lab provided practical experience with a robot control system that demonstrates these concepts in action.

These fundamentals form the backbone of any ROS 2-based robotics application. Understanding these concepts is crucial for developing more complex robotic systems in the subsequent chapters, where we'll apply these principles to perception, control, and intelligence systems.