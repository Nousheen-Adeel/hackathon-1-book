---
sidebar_position: 2
title: "ROS 2 Fundamentals"
---

# ROS 2 Fundamentals

## Weekly Plan
- Day 1-2: Understanding ROS 2 architecture and concepts
- Day 3-4: Creating nodes, topics, and services
- Day 5-7: Advanced concepts - actions, parameters, and launch files

## Learning Objectives
By the end of this chapter, you will:
- Understand the ROS 2 architecture and communication patterns
- Create nodes that communicate via topics, services, and actions
- Use parameters for configuration management
- Create launch files for complex system startup

## ROS 2 Architecture

ROS 2 uses a distributed architecture where multiple processes (nodes) communicate through a publish/subscribe model. The key components are:

- **Nodes**: Individual processes that perform specific functions
- **Topics**: Channels for asynchronous message passing
- **Services**: Synchronous request/response communication
- **Actions**: Goal-based communication with feedback
- **Parameters**: Configuration values that can be changed at runtime

### The DDS Middleware
ROS 2 uses Data Distribution Service (DDS) as its middleware, providing:
- Real-time performance
- Quality of Service (QoS) policies
- Reliable message delivery
- Language independence

## Creating Nodes

### Basic Node Structure
```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')
        # Initialize components here
        self.get_logger().info('My Robot Node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Publishers and Subscribers
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

class TalkerListenerNode(Node):
    def __init__(self):
        super().__init__('talker_listener')

        # Create publisher
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create subscriber
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)

        # Timer to publish messages
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = TalkerListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services

Services provide synchronous request/response communication:

```python
# Service server
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}\n')
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
# Service client
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
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
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

Actions are used for long-running tasks with feedback:

```python
# Action server
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')
        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    rclpy.spin(fibonacci_action_server)
    fibonacci_action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameters

Parameters allow runtime configuration of nodes:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_factor', 0.8)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_factor = self.get_parameter('safety_factor').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Safety factor: {self.safety_factor}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files

Launch files allow starting multiple nodes with specific configurations:

```xml
<!-- robot_system.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                {'max_velocity': 2.0},
                {'safety_factor': 0.9}
            ],
            remappings=[
                ('/cmd_vel', '/my_cmd_vel')
            ]
        ),
        Node(
            package='my_robot_package',
            executable='sensor_processor',
            name='sensor_processor',
            parameters=[
                {'sensor_range': 10.0}
            ]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', 'config/my_robot.rviz']
        )
    ])
```

## Quality of Service (QoS) Settings

QoS settings allow fine-tuning communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class QoSDemoNode(Node):
    def __init__(self):
        super().__init__('qos_demo')

        # Create a QoS profile for reliable communication
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.publisher = self.create_publisher(
            String,
            'qos_chatter',
            qos_profile
        )

        self.subscription = self.create_subscription(
            String,
            'qos_chatter',
            self.listener_callback,
            qos_profile
        )
```

## Weekly Exercises

### Exercise 1: Publisher and Subscriber
1. Create a publisher that sends sensor data (temperature, distance, etc.)
2. Create a subscriber that processes and logs the sensor data
3. Test the communication between nodes

### Exercise 2: Service Implementation
1. Create a service that calculates the distance between two points
2. Implement a client that calls the service with different coordinates
3. Test the service with various inputs

### Exercise 3: Action Server
1. Create an action server that moves a robot to a specified position
2. Implement feedback to show progress
3. Create a client that sends goals and monitors progress

### Mini-Project: Robot Arm Controller
Create a complete robot arm controller with:
- Joint position publisher
- Gripper control service
- Trajectory execution action
- Parameter-based configuration

```python
# robot_arm_controller.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from example_interfaces.srv import SetBool
from example_interfaces.action import FollowJointTrajectory
from rclpy.action import ActionServer
import time

class RobotArmController(Node):
    def __init__(self):
        super().__init__('robot_arm_controller')

        # Declare parameters
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3'])
        self.declare_parameter('max_velocity', 1.0)

        # Joint position publisher
        self.joint_pub = self.create_publisher(Float64MultiArray, '/joint_positions', 10)

        # Gripper service
        self.gripper_service = self.create_service(
            SetBool,
            'control_gripper',
            self.gripper_callback
        )

        # Trajectory action server
        self.trajectory_server = ActionServer(
            self,
            FollowJointTrajectory,
            'follow_joint_trajectory',
            self.execute_trajectory
        )

        self.current_joints = [0.0, 0.0, 0.0]
        self.get_logger().info('Robot Arm Controller initialized')

    def gripper_callback(self, request, response):
        if request.data:
            self.get_logger().info('Gripper closing')
        else:
            self.get_logger().info('Gripper opening')
        response.success = True
        response.message = 'Gripper command executed'
        return response

    def execute_trajectory(self, goal_handle):
        self.get_logger().info('Executing trajectory...')

        for point in goal_handle.request.trajectory.points:
            # Move to joint positions
            joint_msg = Float64MultiArray()
            joint_msg.data = point.positions
            self.joint_pub.publish(joint_msg)
            self.current_joints = list(point.positions)

            # Wait for movement
            time.sleep(0.5)

            # Publish feedback
            # (In a real implementation, you'd check actual joint positions)

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = 0
        return result

def main(args=None):
    rclpy.init(args=args)
    controller = RobotArmController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary
This chapter covered the fundamental concepts of ROS 2, including nodes, topics, services, actions, parameters, and launch files. You've learned how to create and configure these components to build complex robotic systems. The next chapter will focus on robot simulation using Gazebo and Unity.