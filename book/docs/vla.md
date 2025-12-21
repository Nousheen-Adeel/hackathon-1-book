---
sidebar_position: 5
title: "Vision-Language-Action (VLA)"
---

# Vision-Language-Action (VLA)

## Weekly Plan
- Day 1-2: Understanding VLA systems and multimodal AI
- Day 3-4: Implementing speech recognition and natural language processing
- Day 5-7: Creating action planning and execution systems with safety

## Learning Objectives
By the end of this chapter, you will:
- Understand Vision-Language-Action (VLA) systems for robotics
- Implement speech recognition using Whisper for voice-to-action conversion
- Create natural language understanding and task planning systems
- Develop safe action execution with multimodal feedback
- Integrate VLA systems with ROS 2 for robot control

## Vision-Language-Action Overview

Vision-Language-Action (VLA) systems represent a paradigm shift in robotics, enabling robots to understand and execute complex commands expressed in natural language. These systems combine:
- Computer vision for scene understanding
- Natural language processing for command interpretation
- Action planning and execution for physical task completion
- Safety protocols to ensure reliable operation

### Key Components of VLA Systems
- **Perception Module**: Processes visual input to understand the environment
- **Language Module**: Interprets natural language commands
- **Planning Module**: Decomposes high-level commands into executable actions
- **Execution Module**: Executes actions with safety monitoring
- **Feedback Module**: Provides multimodal feedback to users

## Speech Recognition with Whisper

### Setting up Whisper for Voice Commands
```python
# Install required packages
# pip install openai-whisper torch

import whisper
import torch
import numpy as np
import pyaudio
import wave
import threading
import queue

class WhisperVoiceProcessor:
    def __init__(self, model_size="base"):
        # Load Whisper model
        self.model = whisper.load_model(model_size)
        self.audio_queue = queue.Queue()

        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 3

        self.audio = pyaudio.PyAudio()
        self.is_listening = False

    def start_listening(self):
        """Start voice recognition in a separate thread"""
        self.is_listening = True
        listening_thread = threading.Thread(target=self._listen_loop)
        listening_thread.start()

    def _listen_loop(self):
        """Continuous listening loop"""
        while self.is_listening:
            # Record audio
            frames = []

            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = stream.read(self.chunk)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            # Save to temporary WAV file
            temp_filename = "temp_recording.wav"
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Transcribe audio
            result = self.model.transcribe(temp_filename)
            if result["text"].strip():  # Only process non-empty results
                self.process_command(result["text"])

    def process_command(self, text):
        """Process the transcribed command"""
        print(f"Recognized: {text}")
        # Send to command processor
        self.command_callback(text)

    def command_callback(self, command_text):
        """Override this method to handle commands"""
        pass

    def stop_listening(self):
        """Stop the listening process"""
        self.is_listening = False

# Example usage
class RobotVoiceController(WhisperVoiceProcessor):
    def __init__(self):
        super().__init__()
        self.robot_commands = {
            "move forward": self.move_forward,
            "move backward": self.move_backward,
            "turn left": self.turn_left,
            "turn right": self.turn_right,
            "stop": self.stop_robot,
            "pick up object": self.pick_object,
            "place object": self.place_object
        }

    def command_callback(self, command_text):
        """Process robot commands"""
        command_text = command_text.lower().strip()

        # Simple command matching (in practice, use more sophisticated NLP)
        for cmd, action in self.robot_commands.items():
            if cmd in command_text:
                print(f"Executing command: {cmd}")
                action()
                return

        print(f"Unknown command: {command_text}")

    def move_forward(self):
        print("Moving robot forward")
        # Publish ROS message to move robot forward
        self.publish_robot_command("move_forward")

    def move_backward(self):
        print("Moving robot backward")
        self.publish_robot_command("move_backward")

    def turn_left(self):
        print("Turning robot left")
        self.publish_robot_command("turn_left")

    def turn_right(self):
        print("Turning robot right")
        self.publish_robot_command("turn_right")

    def stop_robot(self):
        print("Stopping robot")
        self.publish_robot_command("stop")

    def pick_object(self):
        print("Picking up object")
        self.publish_robot_command("pick_object")

    def place_object(self):
        print("Placing object")
        self.publish_robot_command("place_object")

    def publish_robot_command(self, command):
        """Publish command to ROS system (placeholder)"""
        print(f"Publishing to ROS: {command}")

# Example usage
if __name__ == "__main__":
    controller = RobotVoiceController()
    print("Starting voice recognition...")
    controller.start_listening()

    # Keep the program running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        controller.stop_listening()
        controller.audio.terminate()
```

## Natural Language Understanding and Planning

### LLM-Based Task Planning
```python
import openai
from typing import List, Dict, Any
import json

class LLMPlanner:
    def __init__(self, api_key: str = None):
        if api_key:
            openai.api_key = api_key
        self.action_space = [
            "move_to_location",
            "pick_object",
            "place_object",
            "open_gripper",
            "close_gripper",
            "rotate_gripper",
            "wait",
            "check_condition"
        ]

    def plan_task(self, natural_language_command: str, environment_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert natural language command to a sequence of executable actions
        """
        prompt = f"""
        Given the following natural language command and environment state,
        decompose the command into a sequence of executable robotic actions.

        Environment state: {json.dumps(environment_state)}

        Natural language command: "{natural_language_command}"

        Available actions: {', '.join(self.action_space)}

        Return the action sequence as a JSON list of objects with 'action' and 'parameters' keys.
        Example format:
        [
            {{"action": "move_to_location", "parameters": {{"x": 1.0, "y": 2.0, "z": 0.5}}}},
            {{"action": "pick_object", "parameters": {{"object_id": "red_block"}}}},
            {{"action": "move_to_location", "parameters": {{"x": 3.0, "y": 1.0, "z": 0.5}}}},
            {{"action": "place_object", "parameters": {{"object_id": "red_block"}}}}
        ]

        Action sequence:
        """

        try:
            # Using OpenAI API (replace with your preferred LLM)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            action_sequence = json.loads(response.choices[0].message.content)
            return action_sequence

        except Exception as e:
            print(f"Error in LLM planning: {e}")
            return self.fallback_plan(natural_language_command)

    def fallback_plan(self, command: str) -> List[Dict[str, Any]]:
        """
        Simple fallback planner for when LLM is unavailable
        """
        command_lower = command.lower()

        if "pick" in command_lower or "grasp" in command_lower:
            return [
                {"action": "move_to_location", "parameters": {"x": 1.0, "y": 0.0, "z": 0.0}},
                {"action": "pick_object", "parameters": {"object_id": "unknown"}}
            ]
        elif "place" in command_lower or "put" in command_lower:
            return [
                {"action": "move_to_location", "parameters": {"x": 2.0, "y": 0.0, "z": 0.0}},
                {"action": "place_object", "parameters": {"object_id": "unknown"}}
            ]
        elif "move" in command_lower or "go" in command_lower:
            return [
                {"action": "move_to_location", "parameters": {"x": 1.0, "y": 1.0, "z": 0.0}}
            ]
        else:
            return [
                {"action": "wait", "parameters": {"duration": 1.0}}
            ]

# Example usage
class VLAPlanner:
    def __init__(self):
        self.llm_planner = LLMPlanner()  # Initialize with your API key if needed

    def create_plan(self, command: str, env_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a plan from natural language command"""
        plan = self.llm_planner.plan_task(command, env_state)
        return self.validate_plan(plan)

    def validate_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate the plan for safety and feasibility"""
        validated_plan = []

        for action in plan:
            if self.is_action_safe(action) and self.is_action_feasible(action):
                validated_plan.append(action)
            else:
                print(f"Skipping unsafe or infeasible action: {action}")

        return validated_plan

    def is_action_safe(self, action: Dict[str, Any]) -> bool:
        """Check if action is safe to execute"""
        # Implement safety checks
        action_type = action.get("action", "")

        # Example safety checks
        if action_type == "move_to_location":
            params = action.get("parameters", {})
            x, y, z = params.get("x", 0), params.get("y", 0), params.get("z", 0)

            # Check if coordinates are within safe bounds
            if abs(x) > 10 or abs(y) > 10 or z < 0 or z > 2:
                return False

        return True

    def is_action_feasible(self, action: Dict[str, Any]) -> bool:
        """Check if action is feasible given robot capabilities"""
        # Implement feasibility checks
        return True

# Example usage
if __name__ == "__main__":
    planner = VLAPlanner()

    env_state = {
        "objects": [
            {"id": "red_block", "type": "block", "position": [0.5, 0.5, 0.1]},
            {"id": "blue_block", "type": "block", "position": [1.0, 1.0, 0.1]}
        ],
        "robot_position": [0.0, 0.0, 0.0],
        "gripper_status": "open"
    }

    command = "Pick up the red block and place it on the table at position 2, 2"
    plan = planner.create_plan(command, env_state)

    print("Generated plan:")
    for i, action in enumerate(plan):
        print(f"{i+1}. {action['action']} with params {action['parameters']}")
```

## Multi-Modal Interaction System

### Vision-Based Object Recognition
```python
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import requests
from transformers import DetrImageProcessor, DetrForObjectDetection

class VisionSystem:
    def __init__(self):
        # Initialize DETR object detection model
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.eval()

        # COCO dataset labels for object detection
        self.coco_labels = [
            "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
            "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def detect_objects(self, image_path_or_array):
        """
        Detect objects in an image using DETR
        """
        if isinstance(image_path_or_array, str):
            image = Image.open(image_path_or_array)
        else:
            # Convert numpy array to PIL Image
            image = Image.fromarray(cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detected_objects.append({
                "label": self.coco_labels[label.item()],
                "score": score.item(),
                "bbox": box.tolist()  # [x_min, y_min, x_max, y_max]
            })

        return detected_objects

    def find_object_by_description(self, image, description):
        """
        Find objects matching a textual description
        """
        detected_objects = self.detect_objects(image)

        # Simple matching based on description keywords
        description_lower = description.lower()
        matching_objects = []

        for obj in detected_objects:
            if obj["label"] in description_lower or description_lower in obj["label"]:
                matching_objects.append(obj)

        return matching_objects

# Example usage
vision_system = VisionSystem()

# Example: Detect objects in an image
# objects = vision_system.detect_objects("path/to/image.jpg")
# print("Detected objects:", objects)
```

## ROS 2 Integration for VLA Systems

### VLA Command Execution Node
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import json
import threading

class VLAExecutionNode(Node):
    def __init__(self):
        super().__init__('vla_execution_node')

        # Publishers
        self.cmd_pub = self.create_publisher(String, '/vla_commands', 10)
        self.pose_pub = self.create_publisher(Pose, '/robot_target_pose', 10)
        self.status_pub = self.create_publisher(String, '/vla_status', 10)

        # Subscribers
        self.voice_sub = self.create_subscription(
            String, '/voice_commands', self.voice_command_callback, 10
        )
        self.vision_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.vision_callback, 10
        )

        # Services
        self.execute_srv = self.create_service(
            Trigger, '/execute_vla_plan', self.execute_plan_callback
        )

        # Internal state
        self.bridge = CvBridge()
        self.current_plan = []
        self.is_executing = False
        self.current_image = None

        # Vision system
        self.vision_system = VisionSystem()

        self.get_logger().info('VLA Execution Node initialized')

    def voice_command_callback(self, msg):
        """Process voice commands from speech recognition"""
        command_text = msg.data
        self.get_logger().info(f'Received voice command: {command_text}')

        # Process the command through the VLA pipeline
        self.process_vla_command(command_text)

    def vision_callback(self, msg):
        """Process camera images for vision system"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_vla_command(self, command_text):
        """Process a VLA command through the full pipeline"""
        self.get_logger().info(f'Processing VLA command: {command_text}')

        # Update status
        status_msg = String()
        status_msg.data = f'Processing command: {command_text}'
        self.status_pub.publish(status_msg)

        # Get current environment state (simplified)
        env_state = self.get_environment_state()

        # Create plan using VLA planner
        planner = VLAPlanner()
        plan = planner.create_plan(command_text, env_state)

        if plan:
            self.current_plan = plan
            self.get_logger().info(f'Generated plan with {len(plan)} steps')

            # Execute the plan
            self.execute_plan()
        else:
            self.get_logger().warn('Could not generate plan for command')

    def get_environment_state(self):
        """Get current environment state for planning"""
        # In a real system, this would gather state from multiple sensors
        env_state = {
            "objects": [],
            "robot_position": [0.0, 0.0, 0.0],
            "gripper_status": "open"
        }

        # If we have a current image, detect objects
        if self.current_image is not None:
            try:
                # Convert OpenCV image to format expected by vision system
                detected_objects = self.vision_system.detect_objects(self.current_image)
                env_state["objects"] = detected_objects
            except Exception as e:
                self.get_logger().error(f'Error detecting objects: {e}')

        return env_state

    def execute_plan(self):
        """Execute the current plan step by step"""
        if not self.current_plan or self.is_executing:
            return

        self.is_executing = True

        # Execute each action in the plan
        for i, action in enumerate(self.current_plan):
            self.get_logger().info(f'Executing action {i+1}/{len(self.current_plan)}: {action}')

            success = self.execute_single_action(action)

            if not success:
                self.get_logger().error(f'Action failed: {action}')
                break

        self.is_executing = False
        self.current_plan = []

        # Update status
        status_msg = String()
        status_msg.data = 'Plan execution completed'
        self.status_pub.publish(status_msg)

    def execute_single_action(self, action):
        """Execute a single action from the plan"""
        action_type = action.get('action', '')
        parameters = action.get('parameters', {})

        self.get_logger().info(f'Executing action: {action_type} with params: {parameters}')

        if action_type == 'move_to_location':
            return self.execute_move_to_location(parameters)
        elif action_type == 'pick_object':
            return self.execute_pick_object(parameters)
        elif action_type == 'place_object':
            return self.execute_place_object(parameters)
        elif action_type == 'open_gripper':
            return self.execute_open_gripper()
        elif action_type == 'close_gripper':
            return self.execute_close_gripper()
        elif action_type == 'wait':
            return self.execute_wait(parameters)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def execute_move_to_location(self, params):
        """Execute move to location action"""
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        z = params.get('z', 0.0)

        # Create and publish target pose
        pose_msg = Pose()
        pose_msg.position.x = x
        pose_msg.position.y = y
        pose_msg.position.z = z
        # Set orientation to face forward (simplified)
        pose_msg.orientation.z = 0.0
        pose_msg.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)
        self.get_logger().info(f'Moved to location: ({x}, {y}, {z})')

        return True

    def execute_pick_object(self, params):
        """Execute pick object action"""
        object_id = params.get('object_id', 'unknown')
        self.get_logger().info(f'Picking object: {object_id}')

        # Publish gripper command
        cmd_msg = String()
        cmd_msg.data = f'pick_object_{object_id}'
        self.cmd_pub.publish(cmd_msg)

        return True

    def execute_place_object(self, params):
        """Execute place object action"""
        object_id = params.get('object_id', 'unknown')
        self.get_logger().info(f'Placing object: {object_id}')

        # Publish gripper command
        cmd_msg = String()
        cmd_msg.data = f'place_object_{object_id}'
        self.cmd_pub.publish(cmd_msg)

        return True

    def execute_open_gripper(self):
        """Execute open gripper action"""
        self.get_logger().info('Opening gripper')

        cmd_msg = String()
        cmd_msg.data = 'open_gripper'
        self.cmd_pub.publish(cmd_msg)

        return True

    def execute_close_gripper(self):
        """Execute close gripper action"""
        self.get_logger().info('Closing gripper')

        cmd_msg = String()
        cmd_msg.data = 'close_gripper'
        self.cmd_pub.publish(cmd_msg)

        return True

    def execute_wait(self, params):
        """Execute wait action"""
        duration = params.get('duration', 1.0)
        self.get_logger().info(f'Waiting for {duration} seconds')

        # In a real system, you'd use a timer or action client
        # For simulation, just return immediately
        return True

    def execute_plan_callback(self, request, response):
        """Service callback to execute current plan"""
        if self.current_plan:
            self.execute_plan()
            response.success = True
            response.message = 'Plan executed successfully'
        else:
            response.success = False
            response.message = 'No plan available to execute'

        return response

def main(args=None):
    rclpy.init(args=args)
    node = VLAExecutionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VLA Execution Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Safety Protocols in VLA Systems

### Safety Monitor for VLA Execution
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool, String
from builtin_interfaces.msg import Duration
import threading
import time

class VLASafetyMonitor(Node):
    def __init__(self):
        super().__init__('vla_safety_monitor')

        # Publishers
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.safety_status_pub = self.create_publisher(String, '/safety_status', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        self.vla_status_sub = self.create_subscription(
            String, '/vla_status', self.vla_status_callback, 10
        )

        # Parameters
        self.declare_parameter('safe_distance', 0.5)  # meters
        self.declare_parameter('max_linear_velocity', 0.5)  # m/s
        self.declare_parameter('max_angular_velocity', 0.5)  # rad/s

        self.safe_distance = self.get_parameter('safe_distance').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value

        # Internal state
        self.latest_scan = None
        self.latest_cmd_vel = None
        self.vla_active = False
        self.emergency_stop_active = False
        self.safety_violation = False

        # Safety timer
        self.safety_timer = self.create_timer(0.1, self.safety_check_callback)

        self.get_logger().info('VLA Safety Monitor initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.latest_scan = msg

    def cmd_vel_callback(self, msg):
        """Monitor velocity commands"""
        self.latest_cmd_vel = msg

    def vla_status_callback(self, msg):
        """Monitor VLA system status"""
        self.vla_active = 'executing' in msg.data.lower()

    def safety_check_callback(self):
        """Main safety monitoring loop"""
        if self.emergency_stop_active:
            return  # Already in emergency stop state

        # Check for safety violations
        violations = []

        # Check proximity to obstacles
        if self.latest_scan:
            min_distance = min(self.latest_scan.ranges) if self.latest_scan.ranges else float('inf')
            if min_distance < self.safe_distance:
                violations.append(f'Obstacle too close: {min_distance:.2f}m < {self.safe_distance}m')

        # Check velocity limits
        if self.latest_cmd_vel:
            if abs(self.latest_cmd_vel.linear.x) > self.max_linear_vel:
                violations.append(f'Linear velocity too high: {self.latest_cmd_vel.linear.x:.2f} > {self.max_linear_vel}')
            if abs(self.latest_cmd_vel.angular.z) > self.max_angular_vel:
                violations.append(f'Angular velocity too high: {self.latest_cmd_vel.angular.z:.2f} > {self.max_angular_vel}')

        # If violations detected, trigger emergency stop
        if violations:
            self.safety_violation = True
            self.trigger_emergency_stop(violations)
        else:
            self.safety_violation = False

    def trigger_emergency_stop(self, violations):
        """Trigger emergency stop and log violations"""
        self.get_logger().error(f'Safety violation detected: {", ".join(violations)}')

        # Publish emergency stop
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

        self.emergency_stop_active = True

        # Update safety status
        status_msg = String()
        status_msg.data = f'EMERGENCY_STOP: {", ".join(violations)}'
        self.safety_status_pub.publish(status_msg)

    def reset_safety(self):
        """Reset emergency stop state"""
        self.emergency_stop_active = False
        self.safety_violation = False

        # Publish reset
        stop_msg = Bool()
        stop_msg.data = False
        self.emergency_stop_pub.publish(stop_msg)

        status_msg = String()
        status_msg.data = 'SAFETY_NORMAL'
        self.safety_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VLASafetyMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VLA Safety Monitor')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Weekly Exercises

### Exercise 1: Voice Command Recognition
1. Set up Whisper for speech recognition
2. Create a vocabulary of robot commands
3. Test voice recognition accuracy
4. Integrate with a simple robot simulator

### Exercise 2: Natural Language Processing
1. Implement an LLM-based planner using OpenAI API or open-source alternative
2. Test with various natural language commands
3. Validate the generated action sequences
4. Handle ambiguous or unclear commands

### Exercise 3: Multi-Modal Integration
1. Combine vision and language processing
2. Create a system that can identify objects mentioned in commands
3. Test with different object types and environments
4. Implement safety checks for the integrated system

### Mini-Project: Complete VLA System
Create a complete Vision-Language-Action system:
- Voice recognition for command input
- Vision system for environment understanding
- LLM-based planning for task decomposition
- Safe execution with monitoring
- ROS 2 integration for robot control

```python
# Complete VLA System Integration
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import threading
import time

class CompleteVLASystem(Node):
    def __init__(self):
        super().__init__('complete_vla_system')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/vla_system_status', 10)

        # Subscribers
        self.voice_sub = self.create_subscription(
            String, '/voice_commands', self.voice_callback, 10
        )
        self.vision_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.vision_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # Internal components
        self.bridge = CvBridge()
        self.voice_processor = RobotVoiceController()
        self.planner = VLAPlanner()
        self.vision_system = VisionSystem()
        self.safety_monitor = VLASafetyMonitor()

        # State
        self.current_image = None
        self.latest_scan = None
        self.is_executing = False
        self.active_plan = []

        # Start voice processor
        self.voice_processor.start_listening()

        self.get_logger().info('Complete VLA System initialized')

    def voice_callback(self, msg):
        """Handle voice commands"""
        command = msg.data
        self.get_logger().info(f'Processing voice command: {command}')

        # Update status
        status_msg = String()
        status_msg.data = f'Processing: {command}'
        self.status_pub.publish(status_msg)

        # Get environment state
        env_state = self.get_environment_state()

        # Plan and execute
        plan = self.planner.create_plan(command, env_state)
        if plan:
            self.execute_plan(plan)

    def vision_callback(self, msg):
        """Handle vision data"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Vision error: {e}')

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.latest_scan = msg

    def get_environment_state(self):
        """Get current environment state"""
        env_state = {
            "objects": [],
            "robot_position": [0.0, 0.0, 0.0],
            "gripper_status": "open"
        }

        # Add vision data if available
        if self.current_image is not None:
            try:
                detected_objects = self.vision_system.detect_objects(self.current_image)
                env_state["objects"] = detected_objects
            except Exception as e:
                self.get_logger().error(f'Vision processing error: {e}')

        # Add proximity data
        if self.latest_scan:
            min_distance = min(self.latest_scan.ranges) if self.latest_scan.ranges else float('inf')
            env_state["min_obstacle_distance"] = min_distance

        return env_state

    def execute_plan(self, plan):
        """Execute a planned sequence of actions"""
        if self.is_executing:
            self.get_logger().warn('Plan execution already in progress')
            return

        self.is_executing = True
        self.active_plan = plan

        self.get_logger().info(f'Executing plan with {len(plan)} actions')

        for i, action in enumerate(plan):
            self.get_logger().info(f'Executing action {i+1}/{len(plan)}: {action["action"]}')

            success = self.execute_action(action)
            if not success:
                self.get_logger().error(f'Action failed: {action}')
                break

            # Small delay between actions
            time.sleep(0.1)

        self.is_executing = False
        self.active_plan = []

        # Update status
        status_msg = String()
        status_msg.data = 'Plan completed'
        self.status_pub.publish(status_msg)

    def execute_action(self, action):
        """Execute a single action"""
        action_type = action.get('action', '')
        params = action.get('parameters', {})

        if action_type == 'move_to_location':
            return self.move_to_location(params)
        elif action_type == 'wait':
            duration = params.get('duration', 1.0)
            time.sleep(duration)
            return True
        else:
            # For other actions, publish as command
            cmd_msg = String()
            cmd_msg.data = f'{action_type}_{json.dumps(params)}'
            self.cmd_pub.publish(cmd_msg)
            return True

    def move_to_location(self, params):
        """Move robot to specified location"""
        target_x = params.get('x', 0.0)
        target_y = params.get('y', 0.0)

        # Simple proportional controller
        current_x, current_y = 0.0, 0.0  # In real system, get from odometry

        dx = target_x - current_x
        dy = target_y - current_y
        distance = (dx**2 + dy**2)**0.5

        if distance > 0.1:  # If not close enough
            cmd = Twist()
            cmd.linear.x = min(0.5, distance) * (1 if dx > 0 else -1)  # Simplified
            cmd.angular.z = min(0.5, abs(dy)) * (1 if dy > 0 else -1)  # Simplified

            self.cmd_pub.publish(cmd)
            time.sleep(0.5)  # Move for half a second
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

        return True

def main(args=None):
    rclpy.init(args=args)
    node = CompleteVLASystem()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Complete VLA System')
    finally:
        node.voice_processor.stop_listening()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary
This chapter covered Vision-Language-Action (VLA) systems for robotics, including speech recognition with Whisper, natural language processing with LLMs, multimodal interaction, and safety protocols. You've learned how to create systems that understand natural language commands, perceive their environment visually, and execute complex tasks safely. The next chapter will integrate all these concepts in a comprehensive capstone project.