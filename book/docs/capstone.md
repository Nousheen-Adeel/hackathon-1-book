---
sidebar_position: 6
title: "Capstone Project: Autonomous Humanoid"
---

# Capstone Project: Autonomous Humanoid

## Weekly Plan
- Day 1-2: System architecture and component integration
- Day 3-4: Implementing perception-action loop
- Day 5-7: Creating complete autonomous behaviors and testing

## Learning Objectives
By the end of this chapter, you will:
- Integrate all previously learned technologies (ROS 2, Gazebo, Isaac, VLA)
- Design a complete autonomous humanoid robot system
- Implement perception-action loops for real-time operation
- Create complex autonomous behaviors combining multiple capabilities
- Test and validate the integrated system in simulation and potentially on real hardware

## System Architecture Overview

The autonomous humanoid system combines all the technologies learned in previous chapters into a cohesive architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Autonomous Humanoid System                   │
├─────────────────────────────────────────────────────────────────┤
│  Perception Layer                                               │
│  ├─ Vision Processing (Isaac/CV)                               │
│  ├─ Audio Processing (Whisper)                                 │
│  ├─ Sensor Fusion                                              │
│  └─ Environment Mapping                                        │
├─────────────────────────────────────────────────────────────────┤
│  Cognition Layer                                                │
│  ├─ Natural Language Understanding (LLM)                       │
│  ├─ Task Planning & Reasoning                                  │
│  ├─ Behavior Selection                                         │
│  └─ Decision Making                                            │
├─────────────────────────────────────────────────────────────────┤
│  Control Layer                                                  │
│  ├─ Motion Planning                                            │
│  ├─ Trajectory Generation                                      │
│  ├─ Motor Control                                              │
│  └─ Balance & Locomotion                                       │
├─────────────────────────────────────────────────────────────────┤
│  Safety & Monitoring Layer                                      │
│  ├─ Collision Avoidance                                        │
│  ├─ Emergency Response                                         │
│  ├─ System Health Monitoring                                   │
│  └─ Human Safety Protocols                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Complete System Integration

### Main Humanoid Controller Node
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan, JointState
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
import threading
import time
import json
from collections import deque

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.status_pub = self.create_publisher(String, '/humanoid_status', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.voice_cmd_sub = self.create_subscription(
            String, '/voice_commands', self.voice_command_callback, 10
        )
        self.vision_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.vision_callback, 10
        )

        # Internal state
        self.current_pose = None
        self.current_twist = None
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.latest_scan = None
        self.current_image = None
        self.voice_commands = deque(maxlen=10)
        self.system_status = "IDLE"
        self.active_behavior = "STANDBY"

        # Behavior threads
        self.behavior_thread = None
        self.is_running = True

        # Initialize subsystems
        self.vision_system = VisionSystem()
        self.vla_planner = VLAPlanner()
        self.safety_monitor = VLASafetyMonitor()

        # Create timer for main control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Humanoid Controller initialized')

    def odom_callback(self, msg):
        """Update robot pose and velocity from odometry"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.latest_scan = msg

    def joint_state_callback(self, msg):
        """Update joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]

    def vision_callback(self, msg):
        """Process camera images"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Vision processing error: {e}')

    def voice_command_callback(self, msg):
        """Process voice commands"""
        self.voice_commands.append({
            'command': msg.data,
            'timestamp': self.get_clock().now()
        })

        # Process the command
        self.process_voice_command(msg.data)

    def process_voice_command(self, command):
        """Process a voice command through the full pipeline"""
        self.get_logger().info(f'Processing voice command: {command}')

        # Update system status
        self.system_status = "PROCESSING_COMMAND"
        self.publish_status()

        # Get current environment state
        env_state = self.get_environment_state()

        # Plan the task using VLA system
        plan = self.vla_planner.create_plan(command, env_state)

        if plan:
            self.get_logger().info(f'Generated plan with {len(plan)} steps')
            # Execute the plan in a separate thread to avoid blocking
            execution_thread = threading.Thread(target=self.execute_plan, args=(plan,))
            execution_thread.start()
        else:
            self.get_logger().warn('Could not generate plan for command')
            self.system_status = "IDLE"
            self.publish_status()

    def get_environment_state(self):
        """Get complete environment state for planning"""
        env_state = {
            "robot_pose": self.current_pose,
            "robot_twist": self.current_twist,
            "joint_states": {
                "positions": dict(self.joint_positions),
                "velocities": dict(self.joint_velocities),
                "efforts": dict(self.joint_efforts)
            },
            "objects": [],
            "obstacles": [],
            "navigation_map": None
        }

        # Add vision data
        if self.current_image is not None:
            try:
                detected_objects = self.vision_system.detect_objects(self.current_image)
                env_state["objects"] = detected_objects
            except Exception as e:
                self.get_logger().error(f'Vision processing error: {e}')

        # Add obstacle information from laser scan
        if self.latest_scan:
            obstacles = self.process_scan_for_obstacles(self.latest_scan)
            env_state["obstacles"] = obstacles

        return env_state

    def process_scan_for_obstacles(self, scan_msg):
        """Process laser scan to identify obstacles"""
        obstacles = []
        min_distance = min(scan_msg.ranges) if scan_msg.ranges else float('inf')

        # Simple obstacle detection based on scan
        for i, range_val in enumerate(scan_msg.ranges):
            if 0.1 < range_val < 2.0:  # Valid range
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                obstacles.append({
                    "x": x,
                    "y": y,
                    "distance": range_val,
                    "angle": angle
                })

        return obstacles

    def execute_plan(self, plan):
        """Execute a plan in a separate thread"""
        self.active_behavior = "EXECUTING_PLAN"
        self.system_status = "BUSY"

        for i, action in enumerate(plan):
            if not self.is_running:
                break

            self.get_logger().info(f'Executing action {i+1}/{len(plan)}: {action["action"]}')

            success = self.execute_action(action)

            if not success:
                self.get_logger().error(f'Action failed: {action}')
                break

            # Small delay between actions
            time.sleep(0.1)

        self.active_behavior = "STANDBY"
        self.system_status = "IDLE"
        self.publish_status()

    def execute_action(self, action):
        """Execute a single action from the plan"""
        action_type = action.get('action', '')
        params = action.get('parameters', {})

        self.get_logger().info(f'Executing action: {action_type}')

        if action_type == 'move_to_location':
            return self.execute_move_to_location(params)
        elif action_type == 'move_arm':
            return self.execute_move_arm(params)
        elif action_type == 'pick_object':
            return self.execute_pick_object(params)
        elif action_type == 'place_object':
            return self.execute_place_object(params)
        elif action_type == 'rotate_body':
            return self.execute_rotate_body(params)
        elif action_type == 'wait':
            duration = params.get('duration', 1.0)
            time.sleep(duration)
            return True
        else:
            self.get_logger().warn(f'Unknown action: {action_type}')
            return False

    def execute_move_to_location(self, params):
        """Execute move to location action"""
        target_x = params.get('x', 0.0)
        target_y = params.get('y', 0.0)
        target_z = params.get('z', 0.0)

        # Create navigation goal
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = target_x
        goal.pose.position.y = target_y
        goal.pose.position.z = target_z
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)
        return True

    def execute_move_arm(self, params):
        """Execute arm movement action"""
        joint_positions = params.get('joint_positions', {})

        # Create joint command message
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = list(joint_positions.keys())
        joint_cmd.position = list(joint_positions.values())

        self.joint_cmd_pub.publish(joint_cmd)
        return True

    def execute_pick_object(self, params):
        """Execute pick object action"""
        # In a real system, this would control the gripper
        self.get_logger().info('Executing pick object action')

        # Example: close gripper
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = ['gripper_joint']
        joint_cmd.position = [0.0]  # Closed position

        self.joint_cmd_pub.publish(joint_cmd)
        return True

    def execute_place_object(self, params):
        """Execute place object action"""
        self.get_logger().info('Executing place object action')

        # Example: open gripper
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = ['gripper_joint']
        joint_cmd.position = [0.5]  # Open position

        self.joint_cmd_pub.publish(joint_cmd)
        return True

    def execute_rotate_body(self, params):
        """Execute body rotation action"""
        angle = params.get('angle', 0.0)

        cmd = Twist()
        cmd.angular.z = angle  # Simplified - in reality would be more complex
        self.cmd_vel_pub.publish(cmd)

        time.sleep(1.0)  # Wait for rotation
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        return True

    def control_loop(self):
        """Main control loop for the humanoid"""
        if not self.is_running:
            return

        # Check for safety violations
        if self.safety_monitor.safety_violation:
            self.emergency_stop()
            return

        # Update system status
        self.publish_status()

        # Process any pending voice commands
        while self.voice_commands:
            cmd_data = self.voice_commands.popleft()
            # Process command if it's recent (less than 5 seconds old)
            if (self.get_clock().now() - cmd_data['timestamp']).nanoseconds < 5e9:
                self.process_voice_command(cmd_data['command'])

    def publish_status(self):
        """Publish system status"""
        status_msg = String()
        status_msg.data = f"Status: {self.system_status}, Behavior: {self.active_behavior}"
        self.status_pub.publish(status_msg)

    def emergency_stop(self):
        """Execute emergency stop procedures"""
        self.get_logger().error('EMERGENCY STOP ACTIVATED')

        # Stop all motion
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Stop all joints
        joint_stop = JointState()
        joint_stop.header.stamp = self.get_clock().now().to_msg()
        joint_stop.name = list(self.joint_positions.keys())
        joint_stop.position = [0.0] * len(joint_stop.name)
        self.joint_cmd_pub.publish(joint_stop)

        # Update status
        self.system_status = "EMERGENCY_STOP"
        self.publish_status()

    def destroy_node(self):
        """Clean up before node destruction"""
        self.is_running = False
        if self.behavior_thread:
            self.behavior_thread.join(timeout=1.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Humanoid Controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Perception-Action Loop Implementation

### Real-time Perception System
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import threading
import time
from queue import Queue, Empty

class RealTimePerception(Node):
    def __init__(self):
        super().__init__('realtime_perception')

        # Publishers
        self.perception_pub = self.create_publisher(String, '/perception_output', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.pc_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10
        )

        # Internal components
        self.bridge = CvBridge()
        self.vision_system = VisionSystem()

        # Data queues for processing
        self.image_queue = Queue(maxsize=5)
        self.scan_queue = Queue(maxsize=5)
        self.pc_queue = Queue(maxsize=5)

        # Processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Performance monitoring
        self.processing_times = []
        self.frame_count = 0

        self.get_logger().info('Real-time Perception System initialized')

    def image_callback(self, msg):
        """Handle incoming image messages"""
        try:
            if not self.image_queue.full():
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.image_queue.put((cv_image, timestamp))
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def scan_callback(self, msg):
        """Handle incoming laser scan messages"""
        try:
            if not self.scan_queue.full():
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.scan_queue.put((msg, timestamp))
        except Exception as e:
            self.get_logger().error(f'Scan callback error: {e}')

    def pointcloud_callback(self, msg):
        """Handle incoming point cloud messages"""
        try:
            if not self.pc_queue.full():
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.pc_queue.put((msg, timestamp))
        except Exception as e:
            self.get_logger().error(f'Point cloud callback error: {e}')

    def processing_loop(self):
        """Main processing loop running in separate thread"""
        while rclpy.ok():
            start_time = time.time()

            # Process latest image
            try:
                image_data, timestamp = self.image_queue.get_nowait()
                processed_result = self.process_image_data(image_data, timestamp)
                self.publish_perception_result(processed_result)
            except Empty:
                pass  # No new image to process

            # Process latest scan
            try:
                scan_data, timestamp = self.scan_queue.get_nowait()
                processed_result = self.process_scan_data(scan_data, timestamp)
                self.publish_perception_result(processed_result)
            except Empty:
                pass  # No new scan to process

            # Process latest point cloud
            try:
                pc_data, timestamp = self.pc_queue.get_nowait()
                processed_result = self.process_pointcloud_data(pc_data, timestamp)
                self.publish_perception_result(processed_result)
            except Empty:
                pass  # No new point cloud to process

            # Calculate and store processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Maintain reasonable processing rate
            time.sleep(0.01)  # ~100Hz processing rate

    def process_image_data(self, image, timestamp):
        """Process image data for perception"""
        try:
            # Detect objects in the image
            objects = self.vision_system.detect_objects(image)

            # Create perception result
            result = {
                'type': 'image',
                'timestamp': timestamp,
                'objects': objects,
                'frame_count': self.frame_count
            }

            self.frame_count += 1
            return result
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')
            return None

    def process_scan_data(self, scan_msg, timestamp):
        """Process laser scan data for perception"""
        try:
            # Extract relevant information from scan
            ranges = np.array(scan_msg.ranges)
            valid_ranges = ranges[(ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)]

            # Calculate statistics
            if len(valid_ranges) > 0:
                min_distance = np.min(valid_ranges)
                avg_distance = np.mean(valid_ranges)
            else:
                min_distance = float('inf')
                avg_distance = float('inf')

            result = {
                'type': 'laser_scan',
                'timestamp': timestamp,
                'min_distance': min_distance,
                'avg_distance': avg_distance,
                'valid_points': len(valid_ranges)
            }

            return result
        except Exception as e:
            self.get_logger().error(f'Scan processing error: {e}')
            return None

    def process_pointcloud_data(self, pc_msg, timestamp):
        """Process point cloud data for 3D perception"""
        try:
            # In a real implementation, convert PointCloud2 to numpy array
            # For this example, we'll just return basic info
            result = {
                'type': 'pointcloud',
                'timestamp': timestamp,
                'point_count': len(pc_msg.data) // 16  # Approximate (assuming 16 bytes per point)
            }

            return result
        except Exception as e:
            self.get_logger().error(f'Point cloud processing error: {e}')
            return None

    def publish_perception_result(self, result):
        """Publish perception results"""
        if result:
            result_msg = String()
            result_msg.data = json.dumps(result)
            self.perception_pub.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePerception()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Real-time Perception System')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Autonomous Behaviors

### Behavior Manager for Complex Actions
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import threading
import time
import math

class BehaviorManager(Node):
    def __init__(self):
        super().__init__('behavior_manager')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.behavior_status_pub = self.create_publisher(String, '/behavior_status', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/behavior_commands', self.command_callback, 10
        )

        # Internal state
        self.current_pose = None
        self.current_twist = None
        self.latest_scan = None
        self.current_behavior = "IDLE"
        self.behavior_active = False
        self.behavior_thread = None

        # Register behaviors
        self.behaviors = {
            'explore': self.explore_behavior,
            'follow_wall': self.follow_wall_behavior,
            'navigate_to_goal': self.navigate_to_goal_behavior,
            'avoid_obstacles': self.avoid_obstacles_behavior,
            'patrol': self.patrol_behavior,
            'greet_human': self.greet_human_behavior
        }

        self.get_logger().info('Behavior Manager initialized')

    def odom_callback(self, msg):
        """Update robot pose and velocity"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.latest_scan = msg

    def command_callback(self, msg):
        """Handle behavior commands"""
        command = msg.data
        self.get_logger().info(f'Received behavior command: {command}')

        if command in self.behaviors:
            self.start_behavior(command)
        elif command == 'stop':
            self.stop_behavior()
        else:
            self.get_logger().warn(f'Unknown behavior command: {command}')

    def start_behavior(self, behavior_name):
        """Start a specific behavior"""
        if self.behavior_active:
            self.stop_behavior()

        self.current_behavior = behavior_name
        self.behavior_active = True

        # Start behavior in separate thread
        self.behavior_thread = threading.Thread(
            target=self.behaviors[behavior_name]
        )
        self.behavior_thread.start()

        self.update_status(f'Started behavior: {behavior_name}')

    def stop_behavior(self):
        """Stop the current behavior"""
        self.behavior_active = False

        if self.behavior_thread:
            self.behavior_thread.join(timeout=1.0)

        # Stop robot motion
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        self.current_behavior = "IDLE"
        self.update_status('Behavior stopped')

    def update_status(self, status):
        """Update behavior status"""
        status_msg = String()
        status_msg.data = f'{self.current_behavior}: {status}'
        self.behavior_status_pub.publish(status_msg)

    def explore_behavior(self):
        """Random exploration behavior"""
        self.get_logger().info('Starting exploration behavior')

        while self.behavior_active:
            if not self.latest_scan:
                time.sleep(0.1)
                continue

            # Find safe direction to move
            safe_direction = self.find_safe_direction()

            if safe_direction:
                cmd = Twist()
                cmd.linear.x = 0.3  # Move forward at 0.3 m/s
                cmd.angular.z = safe_direction * 0.2  # Gentle turn
                self.cmd_vel_pub.publish(cmd)
            else:
                # Turn in place to find a clear direction
                cmd = Twist()
                cmd.angular.z = 0.5  # Turn right
                self.cmd_vel_pub.publish(cmd)

            time.sleep(0.1)

        # Stop when behavior ends
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

    def follow_wall_behavior(self):
        """Wall following behavior"""
        self.get_logger().info('Starting wall following behavior')

        target_distance = 0.5  # meters from wall
        kp = 1.0  # Proportional gain

        while self.behavior_active:
            if not self.latest_scan:
                time.sleep(0.1)
                continue

            # Get distance to wall on right side (simplified)
            right_distances = self.latest_scan.ranges[270:360]  # Right side
            if right_distances:
                avg_right_distance = sum(right_distances) / len(right_distances)
                error = avg_right_distance - target_distance
                angular_vel = kp * error

                cmd = Twist()
                cmd.linear.x = 0.2  # Forward speed
                cmd.angular.z = angular_vel  # Turn to maintain distance
                self.cmd_vel_pub.publish(cmd)

            time.sleep(0.1)

        # Stop when behavior ends
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

    def navigate_to_goal_behavior(self):
        """Navigate to a specific goal"""
        self.get_logger().info('Starting navigation to goal behavior')

        # In a real system, this would use a navigation stack
        # For this example, we'll simulate navigation to a fixed goal
        goal_x, goal_y = 5.0, 5.0  # Fixed goal for demonstration

        while self.behavior_active:
            if not self.current_pose:
                time.sleep(0.1)
                continue

            # Calculate direction to goal
            dx = goal_x - self.current_pose.position.x
            dy = goal_y - self.current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance < 0.5:  # Close enough to goal
                self.get_logger().info('Reached goal')
                break

            # Calculate angle to goal
            target_angle = math.atan2(dy, dx)
            current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

            # Simple proportional controller
            angle_diff = target_angle - current_yaw
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            cmd = Twist()
            cmd.linear.x = min(0.5, distance) * 0.5  # Scale speed with distance
            cmd.angular.z = angle_diff * 1.0  # Turn toward goal

            # Check for obstacles before moving forward
            if self.latest_scan and min(self.latest_scan.ranges) < 0.5:
                cmd.linear.x = 0.0  # Stop if obstacle too close

            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

        # Stop when behavior ends
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

    def avoid_obstacles_behavior(self):
        """Obstacle avoidance behavior"""
        self.get_logger().info('Starting obstacle avoidance behavior')

        while self.behavior_active:
            if not self.latest_scan:
                time.sleep(0.1)
                continue

            # Check for obstacles in front
            front_scan = self.latest_scan.ranges[330:30] + self.latest_scan.ranges[330:360]  # Front 60 degrees
            min_front_distance = min(front_scan) if front_scan else float('inf')

            cmd = Twist()

            if min_front_distance < 0.5:  # Obstacle detected
                # Stop and turn away from obstacle
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5 if self.get_clear_side() == 'right' else -0.5
            else:
                # Move forward if clear
                cmd.linear.x = 0.3
                cmd.angular.z = 0.0

            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

        # Stop when behavior ends
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

    def patrol_behavior(self):
        """Patrol between predefined waypoints"""
        self.get_logger().info('Starting patrol behavior')

        # Define patrol waypoints
        waypoints = [
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0)
        ]

        current_waypoint = 0

        while self.behavior_active:
            if not self.current_pose:
                time.sleep(0.1)
                continue

            # Get current waypoint
            target_x, target_y = waypoints[current_waypoint]

            # Calculate distance to waypoint
            dx = target_x - self.current_pose.position.x
            dy = target_y - self.current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance < 0.5:  # Reached waypoint
                current_waypoint = (current_waypoint + 1) % len(waypoints)
                self.get_logger().info(f'Reached waypoint {current_waypoint}, moving to next')
                time.sleep(1.0)  # Brief pause at waypoint
                continue

            # Navigate to waypoint
            target_angle = math.atan2(dy, dx)
            current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

            angle_diff = target_angle - current_yaw
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            cmd = Twist()
            cmd.linear.x = min(0.5, distance) * 0.5
            cmd.angular.z = angle_diff * 1.0

            # Avoid obstacles
            if self.latest_scan and min(self.latest_scan.ranges) < 0.5:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn to avoid

            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

    def greet_human_behavior(self):
        """Human greeting behavior"""
        self.get_logger().info('Starting human greeting behavior')

        # This would involve detecting humans, moving toward them,
        # and performing greeting actions (waving, speaking, etc.)
        # For this example, we'll just turn to face forward

        for _ in range(50):  # Run for 5 seconds
            if not self.behavior_active:
                break

            cmd = Twist()
            cmd.angular.z = 0.0  # Face forward
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

    def find_safe_direction(self):
        """Find a safe direction to move based on scan data"""
        if not self.latest_scan:
            return 0.0

        # Divide scan into sectors
        sector_size = len(self.latest_scan.ranges) // 8  # 8 sectors
        sectors = []

        for i in range(8):
            start_idx = i * sector_size
            end_idx = min((i + 1) * sector_size, len(self.latest_scan.ranges))
            sector_ranges = self.latest_scan.ranges[start_idx:end_idx]
            avg_distance = sum(sector_ranges) / len(sector_ranges) if sector_ranges else 0
            sectors.append(avg_distance)

        # Find the sector with maximum average distance (safest)
        max_idx = sectors.index(max(sectors))

        # Convert sector index to angle (-π to π)
        angle = (max_idx * 2 * math.pi / 8) - math.pi
        return angle

    def get_clear_side(self):
        """Determine which side is clearer for obstacle avoidance"""
        if not self.latest_scan:
            return 'right'

        left_distances = self.latest_scan.ranges[90:180]
        right_distances = self.latest_scan.ranges[180:270]

        left_avg = sum(left_distances) / len(left_distances) if left_distances else 0
        right_avg = sum(right_distances) / len(right_distances) if right_distances else 0

        return 'left' if left_avg > right_avg else 'right'

    def get_yaw_from_quaternion(self, q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Behavior Manager')
    finally:
        node.stop_behavior()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## System Testing and Validation

### Comprehensive Testing Framework
```python
import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import threading
import time

class HumanoidSystemTester(Node):
    def __init__(self):
        super().__init__('humanoid_system_tester')

        # Publishers for testing
        self.test_cmd_pub = self.create_publisher(String, '/test_commands', 10)
        self.test_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers for monitoring
        self.status_sub = self.create_subscription(
            String, '/humanoid_status', self.status_callback, 10
        )
        self.test_result_pub = self.create_publisher(String, '/test_results', 10)

        # Internal state
        self.current_status = ""
        self.test_results = []
        self.test_active = False

        self.get_logger().info('Humanoid System Tester initialized')

    def status_callback(self, msg):
        """Update current status"""
        self.current_status = msg.data

    def run_comprehensive_tests(self):
        """Run all system tests"""
        self.get_logger().info('Starting comprehensive system tests')

        tests = [
            self.test_perception_system,
            self.test_navigation_system,
            self.test_vla_integration,
            self.test_safety_system,
            self.test_behavior_manager
        ]

        results = {}
        for test_func in tests:
            test_name = test_func.__name__
            self.get_logger().info(f'Running test: {test_name}')

            try:
                result = test_func()
                results[test_name] = result
                self.get_logger().info(f'Test {test_name}: {result}')
            except Exception as e:
                results[test_name] = f'FAILED: {str(e)}'
                self.get_logger().error(f'Test {test_name} failed: {str(e)}')

        # Publish summary
        summary = json.dumps(results, indent=2)
        result_msg = String()
        result_msg.data = f"Test Summary:\n{summary}"
        self.test_result_pub.publish(result_msg)

        return results

    def test_perception_system(self):
        """Test perception system functionality"""
        # This would involve checking if perception topics are publishing
        # For this example, we'll simulate the test
        time.sleep(1.0)  # Allow some time for system to stabilize

        # Check if we're receiving perception data
        # In a real test, you'd verify data is being published
        return "PASSED"  # Simulated result

    def test_navigation_system(self):
        """Test navigation system"""
        # Send a simple navigation command
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1.0
        goal.pose.position.y = 1.0
        goal.pose.orientation.w = 1.0

        # In a real test, you'd verify the robot reaches the goal
        return "PASSED"  # Simulated result

    def test_vla_integration(self):
        """Test Vision-Language-Action integration"""
        # Send a voice command through the system
        cmd_msg = String()
        cmd_msg.data = "move forward 1 meter"

        # In a real test, you'd verify the command is processed correctly
        return "PASSED"  # Simulated result

    def test_safety_system(self):
        """Test safety system"""
        # This would involve triggering safety conditions
        # and verifying the system responds appropriately
        return "PASSED"  # Simulated result

    def test_behavior_manager(self):
        """Test behavior manager"""
        # Send behavior commands and verify they execute
        behavior_cmd = String()
        behavior_cmd.data = "explore"

        # In a real test, you'd verify the behavior executes
        return "PASSED"  # Simulated result

def main(args=None):
    rclpy.init(args=args)
    tester = HumanoidSystemTester()

    # Run tests in a separate thread to allow ROS spinning
    def run_tests():
        time.sleep(2.0)  # Wait for system to initialize
        results = tester.run_comprehensive_tests()
        tester.get_logger().info(f'All tests completed: {results}')

    test_thread = threading.Thread(target=run_tests)
    test_thread.start()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Shutting down Humanoid System Tester')
    finally:
        test_thread.join(timeout=2.0)
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch File for Complete System

### Complete System Launch
```xml
<!-- launch/humanoid_system.launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Include Gazebo launch (if using simulation)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ])
    )

    # Humanoid controller node
    humanoid_controller = Node(
        package='my_humanoid_package',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Real-time perception node
    perception_node = Node(
        package='my_humanoid_package',
        executable='realtime_perception',
        name='realtime_perception',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Behavior manager node
    behavior_manager = Node(
        package='my_humanoid_package',
        executable='behavior_manager',
        name='behavior_manager',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # VLA execution node
    vla_node = Node(
        package='my_humanoid_package',
        executable='vla_execution',
        name='vla_execution',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Safety monitor node
    safety_monitor = Node(
        package='my_humanoid_package',
        executable='safety_monitor',
        name='safety_monitor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Voice recognition node (if using Whisper)
    voice_recognition = Node(
        package='my_humanoid_package',
        executable='voice_recognition',
        name='voice_recognition',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Navigation stack (Nav2)
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Return launch description with all components
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        TimerAction(
            period=1.0,
            actions=[gazebo]
        ),
        TimerAction(
            period=3.0,
            actions=[humanoid_controller]
        ),
        TimerAction(
            period=4.0,
            actions=[perception_node]
        ),
        TimerAction(
            period=5.0,
            actions=[behavior_manager]
        ),
        TimerAction(
            period=6.0,
            actions=[vla_node]
        ),
        TimerAction(
            period=7.0,
            actions=[safety_monitor]
        ),
        TimerAction(
            period=8.0,
            actions=[voice_recognition]
        ),
        TimerAction(
            period=10.0,
            actions=[navigation]
        )
    ])
```

## Weekly Exercises

### Exercise 1: System Integration
1. Integrate all components (ROS 2, Gazebo, Isaac, VLA) into a single system
2. Test communication between all subsystems
3. Verify that data flows correctly between components
4. Debug any integration issues

### Exercise 2: Perception-Action Loop
1. Implement a real-time perception system
2. Create feedback loops between perception and action
3. Test system response time and accuracy
4. Optimize for real-time performance

### Exercise 3: Autonomous Behaviors
1. Implement multiple autonomous behaviors
2. Create behavior switching logic
3. Test behaviors in various scenarios
4. Implement safety checks for each behavior

### Mini-Project: Complete Autonomous Humanoid
Create a complete autonomous humanoid system with:
- Integrated perception, planning, and control
- Multiple autonomous behaviors
- Voice and vision-based interaction
- Safety monitoring and emergency response
- Comprehensive testing framework

```python
# Complete Autonomous Humanoid System
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan, JointState
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import threading
import time
import json
from collections import deque

class CompleteAutonomousHumanoid(Node):
    def __init__(self):
        super().__init__('complete_autonomous_humanoid')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.status_pub = self.create_publisher(String, '/complete_system_status', 10)
        self.emergency_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.voice_cmd_sub = self.create_subscription(
            String, '/voice_commands', self.voice_command_callback, 10
        )
        self.vision_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.vision_callback, 10
        )

        # Internal state
        self.current_pose = None
        self.current_twist = None
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.latest_scan = None
        self.current_image = None
        self.voice_commands = deque(maxlen=10)
        self.system_status = "BOOTING"
        self.active_behavior = "SYSTEM_CHECK"
        self.emergency_stop = False
        self.is_operational = False

        # Initialize all subsystems
        self.vision_system = VisionSystem()
        self.vla_planner = VLAPlanner()
        self.safety_monitor = VLASafetyMonitor()
        self.behavior_manager = BehaviorManager()

        # Control loop timer
        self.control_timer = self.create_timer(0.1, self.main_control_loop)

        # Start system initialization
        self.init_timer = self.create_timer(2.0, self.initialize_system)

        self.get_logger().info('Complete Autonomous Humanoid System initialized')

    def initialize_system(self):
        """Initialize the complete system"""
        self.get_logger().info('Initializing complete autonomous humanoid system...')

        # Initialize all components
        init_success = True

        if init_success:
            self.system_status = "READY"
            self.is_operational = True
            self.active_behavior = "IDLE"
            self.get_logger().info('System initialization complete')
        else:
            self.system_status = "INITIALIZATION_FAILED"
            self.get_logger().error('System initialization failed')

        self.publish_status()

    def main_control_loop(self):
        """Main control loop for the complete system"""
        if not self.is_operational or self.emergency_stop:
            return

        # Monitor system health
        self.monitor_system_health()

        # Process incoming commands
        self.process_incoming_commands()

        # Execute active behavior
        self.execute_active_behavior()

        # Publish system status
        self.publish_status()

    def monitor_system_health(self):
        """Monitor overall system health"""
        # Check for safety violations
        if hasattr(self.safety_monitor, 'safety_violation') and self.safety_monitor.safety_violation:
            self.trigger_emergency_stop("Safety violation detected")

        # Check component health
        # (In a real system, you'd check if all nodes are responding)

    def process_incoming_commands(self):
        """Process incoming voice and other commands"""
        # Process voice commands
        while self.voice_commands:
            cmd_data = self.voice_commands.popleft()
            # Process command if it's recent (less than 5 seconds old)
            if (self.get_clock().now() - cmd_data['timestamp']).nanoseconds < 5e9:
                self.process_voice_command(cmd_data['command'])

    def execute_active_behavior(self):
        """Execute the currently active behavior"""
        # In a real system, this would manage the active behavior
        pass

    def publish_status(self):
        """Publish comprehensive system status"""
        status_data = {
            'system_status': self.system_status,
            'active_behavior': self.active_behavior,
            'operational': self.is_operational,
            'emergency_stop': self.emergency_stop,
            'timestamp': self.get_clock().now().nanoseconds
        }

        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)

    def trigger_emergency_stop(self, reason="Unknown"):
        """Trigger emergency stop procedure"""
        self.get_logger().error(f'EMERGENCY STOP: {reason}')

        self.emergency_stop = True
        self.system_status = "EMERGENCY_STOP"

        # Stop all motion
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Stop all joints
        joint_stop = JointState()
        joint_stop.header.stamp = self.get_clock().now().to_msg()
        joint_stop.name = list(self.joint_positions.keys())
        joint_stop.position = [0.0] * len(joint_stop.name)
        self.joint_cmd_pub.publish(joint_stop)

        # Publish emergency stop signal
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_pub.publish(emergency_msg)

        self.publish_status()

    def reset_emergency_stop(self):
        """Reset emergency stop state"""
        self.emergency_stop = False
        self.system_status = "READY"
        self.is_operational = True

        emergency_msg = Bool()
        emergency_msg.data = False
        self.emergency_pub.publish(emergency_msg)

        self.publish_status()

def main(args=None):
    rclpy.init(args=args)
    node = CompleteAutonomousHumanoid()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Complete Autonomous Humanoid System')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This capstone project integrates all the technologies covered in the previous chapters to create a complete autonomous humanoid robot system. The system combines:

1. **ROS 2** for communication and coordination between components
2. **Gazebo** for simulation and testing
3. **NVIDIA Isaac** for AI-powered perception and navigation
4. **Vision-Language-Action (VLA)** systems for natural interaction

The complete system architecture includes:
- Real-time perception and environment understanding
- Natural language processing and task planning
- Autonomous behavior execution
- Safety monitoring and emergency response
- Comprehensive testing and validation

This project demonstrates the practical application of all the concepts learned throughout the book, showing how to combine individual technologies into a cohesive, functional autonomous robot system. The modular design allows for easy extension and modification, making it suitable for various robotic applications.

Students should now have a comprehensive understanding of modern robotics technologies and how to integrate them into complex autonomous systems.