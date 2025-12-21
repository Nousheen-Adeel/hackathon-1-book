---
title: Chapter 13 - Multi-Robot Systems
sidebar_position: 1
---

# Chapter 13: Multi-Robot Systems and Coordination

## Learning Goals

- Understand multi-robot system architectures
- Learn coordination and communication protocols
- Master swarm robotics concepts
- Implement multi-robot communication systems
- Coordinate robot teams for collaborative tasks
- Simulate swarm behaviors in robotics environments

## Introduction to Multi-Robot Systems

Multi-robot systems represent a paradigm where multiple autonomous robots collaborate to accomplish tasks that would be difficult or impossible for a single robot to perform alone. These systems leverage the collective capabilities of multiple agents to achieve greater efficiency, robustness, and scalability than individual robots.

### Advantages of Multi-Robot Systems

1. **Scalability**: Tasks can be distributed across multiple robots
2. **Fault Tolerance**: System continues operating despite individual robot failures
3. **Parallelism**: Multiple tasks can be executed simultaneously
4. **Spatial Distribution**: Coverage of large areas or multiple locations
5. **Specialization**: Different robots can have complementary capabilities
6. **Cost Effectiveness**: Multiple simple robots may be cheaper than one complex robot

### Multi-Robot System Challenges

1. **Communication**: Maintaining reliable communication between robots
2. **Coordination**: Ensuring robots work together effectively without conflicts
3. **Task Allocation**: Efficiently distributing tasks among robots
4. **Localization**: Each robot needs to know its position relative to others
5. **Synchronization**: Coordinating actions across the team
6. **Scalability**: Maintaining performance as team size increases

## Multi-Robot Architectures

### Centralized Architecture

In centralized architectures, a central coordinator makes all decisions for the robot team:

```python
import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio


@dataclass
class RobotState:
    robot_id: str
    position: np.ndarray
    status: str  # 'idle', 'working', 'charging', 'error'
    battery_level: float
    capabilities: List[str]
    tasks_completed: int


@dataclass
class Task:
    task_id: str
    description: str
    location: np.ndarray
    priority: int  # Higher number = higher priority
    assigned_robot: Optional[str] = None
    status: str = 'pending'  # 'pending', 'in_progress', 'completed'


class CentralizedMultiRobotController:
    def __init__(self):
        """Initialize centralized multi-robot controller"""
        self.robots: Dict[str, RobotState] = {}
        self.tasks: Dict[str, Task] = {}
        self.communication_channel = {}
        self.task_queue = []
        self.lock = threading.Lock()

        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self.monitor_robot_states)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def add_robot(self, robot_id: str, initial_position: np.ndarray, capabilities: List[str]):
        """Add a robot to the system"""
        with self.lock:
            self.robots[robot_id] = RobotState(
                robot_id=robot_id,
                position=initial_position,
                status='idle',
                battery_level=1.0,
                capabilities=capabilities,
                tasks_completed=0
            )
            print(f"Added robot {robot_id} to system")

    def add_task(self, task_id: str, description: str, location: np.ndarray, priority: int = 1):
        """Add a task to the system"""
        with self.lock:
            task = Task(
                task_id=task_id,
                description=description,
                location=location,
                priority=priority
            )
            self.tasks[task_id] = task
            self.task_queue.append(task_id)
            print(f"Added task {task_id} to queue")

    def allocate_tasks_centralized(self):
        """Allocate tasks to robots based on availability and capabilities"""
        with self.lock:
            # Sort tasks by priority
            pending_tasks = [tid for tid, task in self.tasks.items() if task.status == 'pending']
            pending_tasks.sort(key=lambda tid: self.tasks[tid].priority, reverse=True)

            for task_id in pending_tasks:
                task = self.tasks[task_id]

                # Find available robot with required capabilities
                best_robot = self.find_best_robot_for_task(task)
                if best_robot:
                    # Assign task to robot
                    task.assigned_robot = best_robot
                    task.status = 'in_progress'

                    # Update robot state
                    self.robots[best_robot].status = 'working'

                    print(f"Assigned task {task_id} to robot {best_robot}")

    def find_best_robot_for_task(self, task: Task) -> Optional[str]:
        """Find the best available robot for a given task"""
        best_robot = None
        best_score = -1

        for robot_id, robot_state in self.robots.items():
            if robot_state.status != 'idle':
                continue

            # Check if robot has required capabilities
            if not any(cap in robot_state.capabilities for cap in ['navigation', 'manipulation']):
                continue

            # Calculate score based on distance and battery
            distance = np.linalg.norm(robot_state.position - task.location)
            battery_factor = robot_state.battery_level
            score = battery_factor * (1.0 / (distance + 1))  # Higher score for closer, more charged robots

            if score > best_score:
                best_score = score
                best_robot = robot_id

        return best_robot

    def monitor_robot_states(self):
        """Monitor robot states and update system accordingly"""
        while self.monitoring_active:
            with self.lock:
                for robot_id, robot_state in self.robots.items():
                    # Check if robot has completed assigned task
                    for task_id, task in self.tasks.items():
                        if (task.assigned_robot == robot_id and
                            task.status == 'in_progress' and
                            self.is_robot_at_task_location(robot_id, task.location)):

                            # Complete task
                            task.status = 'completed'
                            robot_state.status = 'idle'
                            robot_state.tasks_completed += 1
                            task.assigned_robot = None
                            print(f"Robot {robot_id} completed task {task_id}")

            time.sleep(1.0)  # Check every second

    def is_robot_at_task_location(self, robot_id: str, target_location: np.ndarray, tolerance: float = 0.5) -> bool:
        """Check if robot is at task location"""
        robot_state = self.robots[robot_id]
        distance = np.linalg.norm(robot_state.position - target_location)
        return distance <= tolerance

    def get_system_status(self) -> Dict:
        """Get overall system status"""
        with self.lock:
            total_tasks = len(self.tasks)
            completed_tasks = len([t for t in self.tasks.values() if t.status == 'completed'])
            active_robots = len([r for r in self.robots.values() if r.status != 'idle'])

            return {
                'total_robots': len(self.robots),
                'active_robots': active_robots,
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'pending_tasks': total_tasks - completed_tasks,
                'system_efficiency': completed_tasks / max(1, total_tasks) if total_tasks > 0 else 0
            }

    def shutdown(self):
        """Shutdown the controller"""
        self.monitoring_active = False
        self.monitoring_thread.join()


# Example usage
def main():
    controller = CentralizedMultiRobotController()

    # Add robots to system
    controller.add_robot('robot_001', np.array([0.0, 0.0]), ['navigation', 'manipulation'])
    controller.add_robot('robot_002', np.array([5.0, 5.0]), ['navigation', 'sensing'])
    controller.add_robot('robot_003', np.array([10.0, 0.0]), ['navigation', 'manipulation', 'sensing'])

    # Add tasks to system
    controller.add_task('task_001', 'Pick up object at (2, 2)', np.array([2.0, 2.0]), priority=3)
    controller.add_task('task_002', 'Survey area at (8, 8)', np.array([8.0, 8.0]), priority=2)
    controller.add_task('task_003', 'Deliver item to (1, 9)', np.array([1.0, 9.0]), priority=1)

    # Run allocation periodically
    for i in range(10):
        controller.allocate_tasks_centralized()

        status = controller.get_system_status()
        print(f"Step {i+1}: System Status - {status}")

        time.sleep(2.0)

    controller.shutdown()


if __name__ == '__main__':
    main()
```

### Decentralized Architecture

Decentralized systems distribute decision-making among robots:

```python
import numpy as np
import threading
import time
import random
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RobotMessage:
    sender_id: str
    message_type: str  # 'task_claim', 'task_complete', 'status_update', 'coordination'
    content: Dict
    timestamp: float


class DecentralizedRobot:
    def __init__(self, robot_id: str, position: np.ndarray, capabilities: List[str]):
        self.robot_id = robot_id
        self.position = position
        self.capabilities = capabilities
        self.status = 'idle'
        self.assigned_task = None
        self.neighbors = {}  # robot_id -> (position, last_seen)
        self.messages = []  # Received messages queue
        self.known_tasks = {}  # task_id -> task_info
        self.completed_tasks = 0
        self.battery_level = 1.0

        # Communication parameters
        self.communication_range = 10.0
        self.message_queue = []

        # Start communication thread
        self.running = True
        self.comm_thread = threading.Thread(target=self.communication_loop)
        self.comm_thread.daemon = True
        self.comm_thread.start()

    def broadcast_message(self, message: RobotMessage, message_bus):
        """Broadcast message to all robots in range"""
        # Calculate which robots are in range
        in_range_robots = []
        for robot_id, (pos, _) in self.neighbors.items():
            distance = np.linalg.norm(self.position - pos)
            if distance <= self.communication_range:
                in_range_robots.append(robot_id)

        # Send message to all in-range robots
        for robot_id in in_range_robots:
            message_bus.send_message(robot_id, message)

    def communication_loop(self):
        """Handle incoming messages"""
        while self.running:
            # Process incoming messages
            for msg in self.messages[:]:  # Copy list to avoid modification during iteration
                self.handle_message(msg)
                self.messages.remove(msg)

            time.sleep(0.1)

    def handle_message(self, message: RobotMessage):
        """Handle incoming message"""
        if message.message_type == 'task_claim':
            # Another robot claimed a task we might have wanted
            task_id = message.content['task_id']
            if task_id in self.known_tasks:
                del self.known_tasks[task_id]

        elif message.message_type == 'task_complete':
            # Task completed by another robot
            task_id = message.content['task_id']
            if task_id in self.known_tasks:
                del self.known_tasks[task_id]

        elif message.message_type == 'status_update':
            # Update neighbor information
            robot_id = message.sender_id
            position = np.array(message.content['position'])
            self.neighbors[robot_id] = (position, time.time())

        elif message.message_type == 'coordination':
            # Handle coordination requests
            self.handle_coordination_request(message)

    def handle_coordination_request(self, message: RobotMessage):
        """Handle coordination requests from other robots"""
        request_type = message.content.get('request_type', '')

        if request_type == 'avoid_collision':
            # Coordinate to avoid collision
            other_pos = np.array(message.content['other_position'])
            other_vel = np.array(message.content['other_velocity'])

            # Calculate collision avoidance maneuver
            avoidance_vector = self.calculate_avoidance_vector(other_pos, other_vel)

            # Send response
            response = RobotMessage(
                sender_id=self.robot_id,
                message_type='coordination_response',
                content={
                    'request_id': message.content['request_id'],
                    'avoidance_vector': avoidance_vector.tolist()
                },
                timestamp=time.time()
            )

            # Broadcast response
            # In practice, this would be sent to the requesting robot specifically

    def calculate_avoidance_vector(self, other_pos: np.ndarray, other_vel: np.ndarray) -> np.ndarray:
        """Calculate avoidance vector to prevent collision"""
        # Simple collision avoidance: move perpendicular to relative velocity
        relative_pos = self.position - other_pos
        distance = np.linalg.norm(relative_pos)

        if distance < 2.0:  # Collision imminent
            # Calculate perpendicular direction
            direction = np.array([-relative_pos[1], relative_pos[0]])  # Perpendicular vector
            direction = direction / (np.linalg.norm(direction) + 1e-8)  # Normalize

            # Move away from other robot
            return direction * 0.5  # Small movement to avoid collision

        return np.array([0.0, 0.0])  # No avoidance needed

    def evaluate_task(self, task_info: Dict) -> float:
        """Evaluate if this robot should claim a task"""
        task_pos = np.array(task_info['location'])

        # Calculate distance to task
        distance = np.linalg.norm(self.position - task_pos)

        # Calculate score based on distance, battery, and capabilities
        battery_factor = self.battery_level
        distance_factor = 1.0 / (distance + 1)  # Closer = higher score
        capability_match = self.calculate_capability_match(task_info['required_capabilities'])

        score = battery_factor * distance_factor * capability_match

        return score

    def calculate_capability_match(self, required_capabilities: List[str]) -> float:
        """Calculate how well this robot matches required capabilities"""
        matches = sum(1 for req_cap in required_capabilities if req_cap in self.capabilities)
        return matches / len(required_capabilities) if required_capabilities else 1.0

    def claim_task(self, task_id: str, task_info: Dict, message_bus) -> bool:
        """Attempt to claim a task"""
        # Evaluate if this robot should claim the task
        score = self.evaluate_task(task_info)

        # Add some randomness to prevent multiple robots claiming same task
        if score > 0.3 and random.random() > 0.3:  # Threshold + random factor
            # Broadcast claim
            claim_msg = RobotMessage(
                sender_id=self.robot_id,
                message_type='task_claim',
                content={
                    'task_id': task_id,
                    'score': score,
                    'position': self.position.tolist()
                },
                timestamp=time.time()
            )

            self.broadcast_message(claim_msg, message_bus)

            # Wait briefly for other claims (auction-style)
            time.sleep(0.5)

            # If no higher-scoring claims received, accept task
            # In practice, this would check for competing claims
            self.assigned_task = task_id
            self.status = 'working'
            self.known_tasks[task_id] = task_info

            print(f"Robot {self.robot_id} claimed task {task_id}")
            return True

        return False

    def complete_task(self, task_id: str, message_bus):
        """Complete assigned task"""
        if self.assigned_task == task_id:
            self.assigned_task = None
            self.status = 'idle'
            self.completed_tasks += 1
            self.battery_level = max(0.0, self.battery_level - 0.1)  # Task consumes battery

            # Broadcast completion
            completion_msg = RobotMessage(
                sender_id=self.robot_id,
                message_type='task_complete',
                content={'task_id': task_id},
                timestamp=time.time()
            )

            self.broadcast_message(completion_msg, message_bus)

            print(f"Robot {self.robot_id} completed task {task_id}")


class MessageBus:
    """Central message bus for inter-robot communication"""
    def __init__(self):
        self.subscribers = {}  # robot_id -> robot_instance
        self.lock = threading.Lock()

    def register_robot(self, robot_id: str, robot_instance):
        """Register a robot with the message bus"""
        with self.lock:
            self.subscribers[robot_id] = robot_instance

    def send_message(self, recipient_id: str, message: RobotMessage):
        """Send message to specific robot"""
        if recipient_id in self.subscribers:
            self.subscribers[recipient_id].messages.append(message)

    def broadcast_message(self, sender_id: str, message: RobotMessage):
        """Broadcast message to all robots"""
        for robot_id, robot_instance in self.subscribers.items():
            if robot_id != sender_id:  # Don't send to sender
                robot_instance.messages.append(message)


class DecentralizedMultiRobotSystem:
    def __init__(self):
        self.robots: Dict[str, DecentralizedRobot] = {}
        self.message_bus = MessageBus()
        self.global_tasks = {}
        self.running = True

    def add_robot(self, robot_id: str, position: np.ndarray, capabilities: List[str]):
        """Add robot to the system"""
        robot = DecentralizedRobot(robot_id, position, capabilities)
        self.robots[robot_id] = robot
        self.message_bus.register_robot(robot_id, robot)

    def add_task(self, task_id: str, task_info: Dict):
        """Add task to the global task pool"""
        self.global_tasks[task_id] = task_info

        # Notify all robots about the new task
        task_announcement = RobotMessage(
            sender_id='system',
            message_type='task_announcement',
            content=task_info,
            timestamp=time.time()
        )

        self.message_bus.broadcast_message('system', task_announcement)

    def run_coordination_cycle(self):
        """Run one cycle of decentralized coordination"""
        for robot_id, robot in self.robots.items():
            # Update robot's knowledge of other robots
            current_time = time.time()
            robot.neighbors = {
                rid: (pos, last_seen)
                for rid, (pos, last_seen) in robot.neighbors.items()
                if current_time - last_seen < 5.0  # Remove neighbors not seen in 5 seconds
            }

            # Robots can claim tasks if they're idle
            if robot.status == 'idle' and self.global_tasks:
                # Get a random task to consider
                task_id = random.choice(list(self.global_tasks.keys()))
                task_info = self.global_tasks[task_id]

                # Attempt to claim the task
                robot.claim_task(task_id, task_info, self.message_bus)

    def get_system_status(self):
        """Get overall system status"""
        total_tasks = len(self.global_tasks)
        completed_tasks = sum(robot.completed_tasks for robot in self.robots.values())
        active_robots = sum(1 for robot in self.robots.values() if robot.status != 'idle')

        return {
            'total_robots': len(self.robots),
            'active_robots': active_robots,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'system_efficiency': completed_tasks / max(1, total_tasks) if total_tasks > 0 else 0
        }

    def run_simulation(self, steps: int = 100):
        """Run multi-robot simulation"""
        for step in range(steps):
            self.run_coordination_cycle()

            if step % 10 == 0:
                status = self.get_system_status()
                print(f"Step {step}: {status}")

            time.sleep(0.5)  # Slow down simulation

        # Stop all robots
        for robot in self.robots.values():
            robot.running = False


# Example usage
def main():
    # Create decentralized multi-robot system
    system = DecentralizedMultiRobotSystem()

    # Add robots to system
    system.add_robot('robot_001', np.array([0.0, 0.0]), ['navigation', 'manipulation'])
    system.add_robot('robot_002', np.array([5.0, 5.0]), ['navigation', 'sensing'])
    system.add_robot('robot_003', np.array([10.0, 0.0]), ['navigation', 'manipulation', 'sensing'])

    # Add tasks to system
    system.add_task('task_001', {
        'task_id': 'task_001',
        'location': [2.0, 2.0],
        'required_capabilities': ['navigation', 'manipulation'],
        'priority': 3
    })

    system.add_task('task_002', {
        'task_id': 'task_002',
        'location': [8.0, 8.0],
        'required_capabilities': ['navigation', 'sensing'],
        'priority': 2
    })

    system.add_task('task_003', {
        'task_id': 'task_003',
        'location': [1.0, 9.0],
        'required_capabilities': ['navigation', 'manipulation'],
        'priority': 1
    })

    print("Starting decentralized multi-robot simulation...")
    system.run_simulation(steps=50)

    # Final status
    final_status = system.get_system_status()
    print(f"\nFinal system status: {final_status}")


if __name__ == '__main__':
    main()
```

## Communication Protocols

### Robot-to-Robot Communication

```python
import asyncio
import json
import zmq
import threading
from typing import Dict, List, Callable


class RobotCommunicationProtocol:
    def __init__(self, robot_id: str, port: int = 5555):
        """
        Initialize robot communication protocol
        robot_id: Unique identifier for this robot
        port: Port for communication
        """
        self.robot_id = robot_id
        self.port = port
        self.context = zmq.Context()

        # Publisher socket for broadcasting
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{port}")

        # Subscriber socket for receiving
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://localhost:{port}")  # Self-connect for local messages
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

        # Callbacks for different message types
        self.callbacks: Dict[str, List[Callable]] = {
            'task_assignment': [],
            'status_update': [],
            'coordination': [],
            'heartbeat': []
        }

        # Start message processing thread
        self.running = True
        self.message_thread = threading.Thread(target=self._message_loop)
        self.message_thread.daemon = True
        self.message_thread.start()

    def register_callback(self, message_type: str, callback: Callable):
        """Register callback for specific message type"""
        if message_type not in self.callbacks:
            self.callbacks[message_type] = []
        self.callbacks[message_type].append(callback)

    def send_message(self, message_type: str, content: Dict, recipients: List[str] = None):
        """Send message to other robots"""
        message = {
            'sender_id': self.robot_id,
            'message_type': message_type,
            'content': content,
            'timestamp': time.time(),
            'recipients': recipients  # If None, broadcast to all
        }

        serialized_msg = json.dumps(message)
        self.publisher.send_string(serialized_msg)

    def broadcast_task_assignment(self, task_id: str, target_robot: str, location: List[float]):
        """Broadcast task assignment"""
        content = {
            'task_id': task_id,
            'target_robot': target_robot,
            'location': location
        }
        self.send_message('task_assignment', content)

    def broadcast_status_update(self, status: str, position: List[float], battery: float):
        """Broadcast status update"""
        content = {
            'status': status,
            'position': position,
            'battery': battery,
            'robot_id': self.robot_id
        }
        self.send_message('status_update', content)

    def broadcast_coordination_request(self, request_type: str, data: Dict):
        """Broadcast coordination request"""
        content = {
            'request_type': request_type,
            'data': data
        }
        self.send_message('coordination', content)

    def _message_loop(self):
        """Internal message processing loop"""
        while self.running:
            try:
                # Receive message
                message_json = self.subscriber.recv_string(flags=zmq.NOBLOCK)
                message = json.loads(message_json)

                # Check if this message is for us or broadcast
                recipients = message.get('recipients')
                if recipients is None or self.robot_id in recipients or self.robot_id == message['sender_id']:
                    # Process message based on type
                    msg_type = message['message_type']
                    if msg_type in self.callbacks:
                        for callback in self.callbacks[msg_type]:
                            try:
                                callback(message)
                            except Exception as e:
                                print(f"Error in callback: {e}")

            except zmq.Again:
                # No message available, sleep briefly
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in message loop: {e}")
                time.sleep(0.1)

    def shutdown(self):
        """Shutdown communication protocol"""
        self.running = False
        if self.message_thread.is_alive():
            self.message_thread.join()
        self.publisher.close()
        self.subscriber.close()
        self.context.term()


class TaskAllocationProtocol:
    def __init__(self, comm_protocol: RobotCommunicationProtocol):
        self.comm_protocol = comm_protocol
        self.available_tasks = {}
        self.assigned_tasks = {}
        self.robot_capabilities = {}  # robot_id -> capabilities

        # Register message handlers
        self.comm_protocol.register_callback('task_announcement', self.handle_task_announcement)
        self.comm_protocol.register_callback('task_bid', self.handle_task_bid)
        self.comm_protocol.register_callback('task_assignment', self.handle_task_assignment)

    def announce_task(self, task_id: str, task_details: Dict):
        """Announce a new task to all robots"""
        self.available_tasks[task_id] = task_details

        content = {
            'task_id': task_id,
            'task_details': task_details
        }

        self.comm_protocol.send_message('task_announcement', content)

    def handle_task_announcement(self, message: Dict):
        """Handle task announcement from other robot"""
        content = message['content']
        task_id = content['task_id']
        task_details = content['task_details']

        self.available_tasks[task_id] = task_details
        print(f"Robot {self.comm_protocol.robot_id} received task announcement: {task_id}")

    def submit_bid(self, task_id: str, bid_value: float):
        """Submit bid for a task"""
        if task_id in self.available_tasks:
            content = {
                'task_id': task_id,
                'bid_value': bid_value,
                'bidder_id': self.comm_protocol.robot_id
            }

            self.comm_protocol.send_message('task_bid', content)

    def handle_task_bid(self, message: Dict):
        """Handle task bid from another robot"""
        content = message['content']
        task_id = content['task_id']
        bid_value = content['bid_value']
        bidder_id = content['bidder_id']

        # In a real auction system, this would track bids and determine winner
        print(f"Received bid from {bidder_id} for task {task_id}: {bid_value}")

    def assign_task(self, task_id: str, robot_id: str):
        """Assign task to specific robot"""
        if task_id in self.available_tasks:
            content = {
                'task_id': task_id,
                'assigned_robot': robot_id
            }

            self.comm_protocol.send_message('task_assignment', content, recipients=[robot_id])

            # Update local state
            self.assigned_tasks[task_id] = robot_id
            if task_id in self.available_tasks:
                del self.available_tasks[task_id]

    def handle_task_assignment(self, message: Dict):
        """Handle task assignment message"""
        content = message['content']
        task_id = content['task_id']
        assigned_robot = content['assigned_robot']

        if self.comm_protocol.robot_id == assigned_robot:
            print(f"Task {task_id} assigned to me!")
            # Robot would start working on the task
        else:
            print(f"Task {task_id} assigned to {assigned_robot}")


# Example usage
def robot_communication_example():
    """Example of robot communication"""

    # Create communication protocols for multiple robots
    robot1_comm = RobotCommunicationProtocol('robot_001', 5555)
    robot2_comm = RobotCommunicationProtocol('robot_002', 5556)
    robot3_comm = RobotCommunicationProtocol('robot_003', 5557)

    # Create task allocation protocols
    ta1 = TaskAllocationProtocol(robot1_comm)
    ta2 = TaskAllocationProtocol(robot2_comm)
    ta3 = TaskAllocationProtocol(robot3_comm)

    # Robot 1 announces a task
    task_details = {
        'location': [5.0, 5.0],
        'type': 'pickup',
        'priority': 1
    }
    ta1.announce_task('task_001', task_details)

    # Simulate some time for message propagation
    time.sleep(1)

    # Robots submit bids for the task
    ta2.submit_bid('task_001', 0.8)  # Robot 2 bids 0.8
    ta3.submit_bid('task_001', 0.6)  # Robot 3 bids 0.6

    # Simulate task assignment (in real system, auctioneer would assign)
    ta1.assign_task('task_001', 'robot_002')  # Assign to highest bidder

    # Send status updates
    robot1_comm.broadcast_status_update('working', [0.0, 0.0], 0.85)
    robot2_comm.broadcast_status_update('traveling', [2.5, 2.5], 0.90)
    robot3_comm.broadcast_status_update('idle', [10.0, 10.0], 1.0)

    # Simulate coordination request
    coord_data = {
        'request_type': 'avoid_collision',
        'target_location': [5.0, 5.0],
        'expected_arrival': time.time() + 10.0
    }
    robot2_comm.broadcast_coordination_request('avoid_collision', coord_data)

    # Let system run briefly
    time.sleep(5)

    # Shutdown
    robot1_comm.shutdown()
    robot2_comm.shutdown()
    robot3_comm.shutdown()

    print("Communication example completed")


if __name__ == '__main__':
    robot_communication_example()
```

## Swarm Robotics

### Collective Behavior Algorithms

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random


class SwarmRobot:
    def __init__(self, position, velocity, robot_id):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.robot_id = robot_id
        self.max_speed = 2.0
        self.perception_radius = 5.0
        self.separation_distance = 1.5
        self.alignment_weight = 0.05
        self.cohesion_weight = 0.05
        self.separation_weight = 0.1
        self.avoidance_weight = 0.1

    def update(self, neighbors, targets, obstacles):
        """Update robot position based on swarm behavior rules"""
        # Initialize forces
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        separation = np.zeros(2)
        avoidance = np.zeros(2)

        if neighbors:
            # Alignment: steer towards average heading of neighbors
            avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
            alignment = (avg_velocity - self.velocity) * self.alignment_weight

            # Cohesion: steer towards average position of neighbors
            avg_position = np.mean([n.position for n in neighbors], axis=0)
            cohesion = (avg_position - self.position) * self.cohesion_weight

            # Separation: steer to avoid crowding neighbors
            separation_vec = np.zeros(2)
            for neighbor in neighbors:
                diff = self.position - neighbor.position
                distance = np.linalg.norm(diff)
                if distance > 0 and distance < self.separation_distance:
                    separation_vec += diff / distance / distance  # Weight by inverse square
            separation = separation_vec * self.separation_weight

        # Avoid obstacles
        for obstacle in obstacles:
            diff = self.position - obstacle
            distance = np.linalg.norm(diff)
            if distance < 3.0:  # Avoidance radius
                avoidance += (diff / distance) * (1.0 / distance) * self.avoidance_weight

        # Seek targets
        target_force = np.zeros(2)
        if targets:
            closest_target = min(targets, key=lambda t: np.linalg.norm(self.position - t))
            target_dir = (closest_target - self.position)
            distance_to_target = np.linalg.norm(target_dir)

            if distance_to_target > 0.5:  # Don't move if very close
                target_force = (target_dir / distance_to_target) * 0.1

        # Apply forces
        self.velocity += alignment + cohesion + separation + avoidance + target_force

        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        # Update position
        self.position += self.velocity

    def get_neighbors(self, all_robots):
        """Get neighboring robots within perception radius"""
        neighbors = []
        for robot in all_robots:
            if robot.robot_id != self.robot_id:
                distance = np.linalg.norm(self.position - robot.position)
                if distance <= self.perception_radius:
                    neighbors.append(robot)
        return neighbors


class SwarmSystem:
    def __init__(self, num_robots=10, world_size=20):
        self.num_robots = num_robots
        self.world_size = world_size
        self.robots = []
        self.targets = []
        self.obstacles = []

        # Initialize robots randomly
        for i in range(num_robots):
            pos = np.random.uniform(0, world_size, 2)
            vel = np.random.uniform(-1, 1, 2)
            robot = SwarmRobot(pos, vel, f'robot_{i:03d}')
            self.robots.append(robot)

    def add_target(self, position):
        """Add a target for the swarm to reach"""
        self.targets.append(np.array(position))

    def add_obstacle(self, position):
        """Add an obstacle to avoid"""
        self.obstacles.append(np.array(position))

    def update_swarm(self):
        """Update all robots in the swarm"""
        for robot in self.robots:
            neighbors = robot.get_neighbors(self.robots)
            robot.update(neighbors, self.targets, self.obstacles)

        # Handle world boundaries (bounce off edges)
        for robot in self.robots:
            if robot.position[0] < 0:
                robot.position[0] = 0
                robot.velocity[0] *= -0.5  # Dampen bounce
            elif robot.position[0] > self.world_size:
                robot.position[0] = self.world_size
                robot.velocity[0] *= -0.5

            if robot.position[1] < 0:
                robot.position[1] = 0
                robot.velocity[1] *= -0.5
            elif robot.position[1] > self.world_size:
                robot.position[1] = self.world_size
                robot.velocity[1] *= -0.5

    def get_positions(self):
        """Get all robot positions for visualization"""
        return [robot.position for robot in self.robots]

    def animate_swarm(self, steps=500):
        """Animate the swarm behavior"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect('equal')
        ax.set_title('Swarm Robotics Simulation')

        # Initialize scatter plot
        positions = np.array(self.get_positions())
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=50)

        # Plot targets
        if self.targets:
            target_positions = np.array(self.targets)
            ax.scatter(target_positions[:, 0], target_positions[:, 1], c='red', s=100, marker='*')

        # Plot obstacles
        if self.obstacles:
            obstacle_positions = np.array(self.obstacles)
            ax.scatter(obstacle_positions[:, 0], obstacle_positions[:, 1], c='black', s=80, marker='s')

        def update(frame):
            self.update_swarm()
            positions = np.array(self.get_positions())
            scatter.set_offsets(positions)
            return scatter,

        anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=True, repeat=True)
        plt.show()
        return anim


def create_swarm_demo():
    """Create and demonstrate swarm behavior"""
    # Create swarm system
    swarm = SwarmSystem(num_robots=20, world_size=20)

    # Add targets and obstacles
    swarm.add_target([15, 15])
    swarm.add_target([5, 15])
    swarm.add_obstacle([10, 10])
    swarm.add_obstacle([8, 12])

    print("Starting swarm simulation...")
    print(f"Swarm with {swarm.num_robots} robots")
    print("Targets: [15, 15], [5, 15]")
    print("Obstacles: [10, 10], [8, 12]")

    # Run animation
    animation = swarm.animate_swarm(steps=500)

    return swarm, animation


if __name__ == '__main__':
    swarm_system, anim = create_swarm_demo()
```

## Task Allocation and Coordination

### Auction-Based Task Allocation

```python
import heapq
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import random


@dataclass
class Task:
    task_id: str
    location: Tuple[float, float]
    value: float  # Reward for completing task
    deadline: float  # Time by which task must be completed
    required_capabilities: List[str]
    priority: int = 1  # Higher number = higher priority

    def distance_to(self, position: Tuple[float, float]) -> float:
        """Calculate distance from position to task location"""
        return ((self.location[0] - position[0])**2 + (self.location[1] - position[1])**2)**0.5


@dataclass
class Robot:
    robot_id: str
    position: Tuple[float, float]
    capabilities: List[str]
    battery_level: float = 1.0
    current_task: str = None
    tasks_completed: int = 0

    def can_perform_task(self, task: Task) -> bool:
        """Check if robot can perform the task based on capabilities"""
        return all(cap in self.capabilities for cap in task.required_capabilities)

    def calculate_utility(self, task: Task) -> float:
        """Calculate utility of task for this robot"""
        if not self.can_perform_task(task):
            return -float('inf')  # Cannot perform task

        # Calculate utility based on value, distance, and battery
        distance_factor = 1.0 / (task.distance_to(self.position) + 1.0)
        battery_factor = self.battery_level
        value_factor = task.value
        priority_factor = task.priority

        # Utility = value * priority * battery * distance_factor
        utility = value_factor * priority_factor * battery_factor * distance_factor

        # Penalize if task might not be completed on time
        estimated_time = task.distance_to(self.position) / 1.0  # Assuming speed of 1.0
        time_remaining = task.deadline - time.time()
        if time_remaining < estimated_time:
            utility *= 0.1  # Heavy penalty for likely missed deadline

        return utility


class AuctionBasedTaskAllocator:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.robots: Dict[str, Robot] = {}
        self.assigned_tasks: Dict[str, str] = {}  # task_id -> robot_id
        self.completed_tasks: List[str] = []

    def add_task(self, task: Task):
        """Add a task to the system"""
        self.tasks[task.task_id] = task
        print(f"Added task {task.task_id} at {task.location} with value {task.value}")

    def add_robot(self, robot: Robot):
        """Add a robot to the system"""
        self.robots[robot.robot_id] = robot
        print(f"Added robot {robot.robot_id} at {robot.position}")

    def run_auction(self) -> Dict[str, str]:
        """Run auction-based task allocation"""
        assignments = {}

        # For each task, run an auction
        for task_id, task in self.tasks.items():
            if task_id in self.assigned_tasks or task_id in self.completed_tasks:
                continue  # Skip already assigned/completed tasks

            # Get bids from all eligible robots
            bids = []
            for robot_id, robot in self.robots.items():
                if robot.current_task is None:  # Only idle robots can bid
                    utility = robot.calculate_utility(task)
                    if utility > -float('inf'):  # Robot can perform task
                        bids.append((utility, robot_id))

            if bids:
                # Sort by utility (highest first)
                bids.sort(reverse=True)
                winning_robot_id = bids[0][1]
                winning_utility = bids[0][0]

                assignments[task_id] = winning_robot_id
                print(f"Task {task_id} assigned to robot {winning_robot_id} (utility: {winning_utility:.2f})")

        return assignments

    def update_assignments(self, new_assignments: Dict[str, str]):
        """Update task assignments"""
        for task_id, robot_id in new_assignments.items():
            if task_id in self.tasks and robot_id in self.robots:
                # Update robot state
                self.robots[robot_id].current_task = task_id
                # Update assignment
                self.assigned_tasks[task_id] = robot_id

    def complete_task(self, task_id: str, robot_id: str):
        """Mark task as completed"""
        if task_id in self.assigned_tasks:
            # Update robot state
            self.robots[robot_id].current_task = None
            self.robots[robot_id].tasks_completed += 1
            self.robots[robot_id].battery_level = max(0.0, self.robots[robot_id].battery_level - 0.1)

            # Update task state
            del self.assigned_tasks[task_id]
            self.completed_tasks.append(task_id)

            print(f"Task {task_id} completed by robot {robot_id}")

    def get_system_status(self) -> Dict:
        """Get overall system status"""
        idle_robots = sum(1 for robot in self.robots.values() if robot.current_task is None)
        active_robots = len(self.robots) - idle_robots
        pending_tasks = len([t for t in self.tasks.keys() if t not in self.assigned_tasks and t not in self.completed_tasks])

        return {
            'total_robots': len(self.robots),
            'idle_robots': idle_robots,
            'active_robots': active_robots,
            'total_tasks': len(self.tasks),
            'pending_tasks': pending_tasks,
            'completed_tasks': len(self.completed_tasks)
        }


class MarketBasedTaskAllocator:
    def __init__(self):
        self.auction_allocator = AuctionBasedTaskAllocator()
        self.task_prices = {}  # task_id -> current price
        self.bid_history = {}  # task_id -> list of (time, price, winning_robot)

    def run_market_based_allocation(self, iterations: int = 5):
        """Run market-based task allocation with price adjustment"""
        for iteration in range(iterations):
            print(f"\n--- Allocation Iteration {iteration + 1} ---")

            # Run auction
            assignments = self.auction_allocator.run_auction()

            # Update assignments
            self.auction_allocator.update_assignments(assignments)

            # Update prices based on competition
            for task_id in assignments.keys():
                if task_id not in self.task_prices:
                    self.task_prices[task_id] = 1.0
                else:
                    # Increase price for competitive tasks
                    self.task_prices[task_id] *= 1.1

            # Print current status
            status = self.auction_allocator.get_system_status()
            print(f"System Status: {status}")

            # Simulate task completion
            for task_id, robot_id in list(self.auction_allocator.assigned_tasks.items()):
                # Simulate completion with some probability
                if random.random() < 0.3:  # 30% chance of completion per iteration
                    self.auction_allocator.complete_task(task_id, robot_id)
                    if task_id in self.task_prices:
                        del self.task_prices[task_id]

            time.sleep(0.5)  # Brief pause between iterations


def main():
    # Create task allocator
    allocator = MarketBasedTaskAllocator()

    # Add robots
    allocator.auction_allocator.add_robot(Robot('robot_001', (0, 0), ['navigation', 'manipulation']))
    allocator.auction_allocator.add_robot(Robot('robot_002', (10, 10), ['navigation', 'sensing']))
    allocator.auction_allocator.add_robot(Robot('robot_003', (5, 5), ['navigation', 'manipulation', 'sensing']))

    # Add tasks
    allocator.auction_allocator.add_task(Task(
        'task_001', (8, 8), value=10.0, deadline=time.time() + 30.0,
        required_capabilities=['navigation', 'manipulation'], priority=3
    ))
    allocator.auction_allocator.add_task(Task(
        'task_002', (2, 2), value=8.0, deadline=time.time() + 25.0,
        required_capabilities=['navigation', 'sensing'], priority=2
    ))
    allocator.auction_allocator.add_task(Task(
        'task_003', (15, 5), value=12.0, deadline=time.time() + 40.0,
        required_capabilities=['navigation', 'manipulation'], priority=1
    ))
    allocator.auction_allocator.add_task(Task(
        'task_004', (1, 15), value=6.0, deadline=time.time() + 20.0,
        required_capabilities=['navigation'], priority=2
    ))

    print("Starting market-based task allocation...")
    allocator.run_market_based_allocation(iterations=10)

    # Final status
    final_status = allocator.auction_allocator.get_system_status()
    print(f"\nFinal system status: {final_status}")


if __name__ == '__main__':
    main()
```

## Coordination Algorithms

### Consensus-Based Coordination

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ConsensusRobot:
    def __init__(self, robot_id, initial_state, neighbors):
        """
        Initialize consensus robot
        robot_id: Unique identifier
        initial_state: Initial state value (scalar or vector)
        neighbors: List of neighboring robot IDs
        """
        self.robot_id = robot_id
        self.state = np.array(initial_state)
        self.neighbors = neighbors
        self.neighbors_states = {}  # neighbor_id -> neighbor_state
        self.consensus_value = self.state.copy()
        self.communication_weights = {}  # neighbor_id -> weight

        # Initialize equal weights for all neighbors
        weight = 1.0 / (len(neighbors) + 1)  # +1 for self
        for neighbor_id in neighbors:
            self.communication_weights[neighbor_id] = weight
        self.self_weight = 1.0 - sum(self.communication_weights.values())

    def update_consensus(self):
        """Update consensus value using weighted average of neighbors"""
        # Weighted sum of neighbors' states
        weighted_sum = self.self_weight * self.state

        for neighbor_id in self.neighbors:
            if neighbor_id in self.neighbors_states:
                neighbor_state = self.neighbors_states[neighbor_id]
                weight = self.communication_weights[neighbor_id]
                weighted_sum += weight * neighbor_state

        # Update consensus value
        self.consensus_value = weighted_sum

    def receive_neighbor_state(self, neighbor_id, neighbor_state):
        """Receive state from neighbor"""
        self.neighbors_states[neighbor_id] = np.array(neighbor_state)

    def get_consensus_value(self):
        """Get current consensus value"""
        return self.consensus_value.copy()


class ConsensusNetwork:
    def __init__(self, num_robots):
        self.num_robots = num_robots
        self.robots = {}
        self.connections = {}  # robot_id -> list of neighbors

        # Create robots with random initial states
        for i in range(num_robots):
            robot_id = f'robot_{i:02d}'
            initial_state = np.random.uniform(-10, 10, 1)  # 1D consensus for simplicity
            self.robots[robot_id] = ConsensusRobot(robot_id, initial_state, [])

    def create_ring_topology(self):
        """Create ring topology where each robot connects to adjacent robots"""
        for i in range(self.num_robots):
            robot_id = f'robot_{i:02d}'
            neighbors = []

            # Connect to previous and next robot (ring)
            prev_idx = (i - 1) % self.num_robots
            next_idx = (i + 1) % self.num_robots

            neighbors.append(f'robot_{prev_idx:02d}')
            neighbors.append(f'robot_{next_idx:02d}')

            self.connections[robot_id] = neighbors
            self.robots[robot_id].neighbors = neighbors

            # Update neighbor weights
            weight = 1.0 / (len(neighbors) + 1)
            for neighbor_id in neighbors:
                self.robots[robot_id].communication_weights[neighbor_id] = weight
            self.robots[robot_id].self_weight = 1.0 - sum(self.robots[robot_id].communication_weights.values())

    def create_random_topology(self, connection_probability=0.3):
        """Create random topology"""
        for i in range(self.num_robots):
            robot_id = f'robot_{i:02d}'
            neighbors = []

            for j in range(self.num_robots):
                if i != j and random.random() < connection_probability:
                    neighbors.append(f'robot_{j:02d}')

            self.connections[robot_id] = neighbors
            self.robots[robot_id].neighbors = neighbors

            # Update neighbor weights
            weight = 1.0 / (len(neighbors) + 1)
            for neighbor_id in neighbors:
                self.robots[robot_id].communication_weights[neighbor_id] = weight
            self.robots[robot_id].self_weight = 1.0 - sum(self.robots[robot_id].communication_weights.values())

    def update_network(self):
        """Update all robots in the network"""
        # First, robots broadcast their states
        for robot_id, robot in self.robots.items():
            for neighbor_id in robot.neighbors:
                if neighbor_id in self.robots:
                    self.robots[neighbor_id].receive_neighbor_state(robot_id, robot.state)

        # Then, robots update their consensus values
        for robot in self.robots.values():
            robot.update_consensus()

        # Finally, update robot states (for dynamic systems)
        for robot in self.robots.values():
            robot.state = robot.consensus_value.copy()

    def get_states(self):
        """Get current states of all robots"""
        return [self.robots[f'robot_{i:02d}'].state[0] for i in range(self.num_robots)]

    def get_consensus_values(self):
        """Get consensus values of all robots"""
        return [self.robots[f'robot_{i:02d}'].consensus_value[0] for i in range(self.num_robots)]

    def animate_consensus(self, steps=100):
        """Animate consensus convergence"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Initialize plots
        x_vals = range(self.num_robots)
        initial_states = self.get_states()

        bars1 = ax1.bar(x_vals, initial_states)
        ax1.set_title('Robot States Over Time')
        ax1.set_ylabel('State Value')
        ax1.set_ylim(min(initial_states) - 1, max(initial_states) + 1)

        consensus_vals = self.get_consensus_values()
        bars2 = ax2.bar(x_vals, consensus_vals)
        ax2.set_title('Consensus Values')
        ax2.set_xlabel('Robot ID')
        ax2.set_ylabel('Consensus Value')
        ax2.set_ylim(min(consensus_vals) - 1, max(consensus_vals) + 1)

        def update(frame):
            self.update_network()

            # Update state bars
            states = self.get_states()
            for bar, height in zip(bars1, states):
                bar.set_height(height)

            # Update consensus bars
            consensus_vals = self.get_consensus_values()
            for bar, height in zip(bars2, consensus_vals):
                bar.set_height(height)

            ax1.set_title(f'Robot States Over Time (Step {frame})')
            ax2.set_title(f'Consensus Values (Step {frame})')

            return list(bars1) + list(bars2)

        anim = FuncAnimation(fig, update, frames=steps, interval=100, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()
        return anim


def main():
    print("Creating consensus network simulation...")

    # Create network with 8 robots
    network = ConsensusNetwork(8)
    network.create_ring_topology()

    print(f"Network topology: {network.connections}")

    # Print initial states
    initial_states = network.get_states()
    print(f"Initial states: {initial_states}")
    print(f"Initial average: {np.mean(initial_states):.2f}")

    # Run animation
    animation = network.animate_consensus(steps=200)

    # Show final states
    final_states = network.get_states()
    print(f"Final states: {final_states}")
    print(f"Final average: {np.mean(final_states):.2f}")

    return network, animation


if __name__ == '__main__':
    network, anim = main()
```

## ROS 2 Multi-Robot Coordination

### Multi-Robot Communication with ROS 2

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan
import json
import numpy as np


class MultiRobotCoordinator(Node):
    def __init__(self):
        super().__init__('multi_robot_coordinator')

        # Robot ID (should be set as parameter)
        self.declare_parameter('robot_id', 'robot_001')
        self.robot_id = self.get_parameter('robot_id').value

        # Publishers
        self.coordination_pub = self.create_publisher(String, f'/{self.robot_id}/coordination', 10)
        self.status_pub = self.create_publisher(String, f'/{self.robot_id}/status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.robot_id}/cmd_vel', 10)

        # Subscribers
        self.coordination_sub = self.create_subscription(
            String,
            '/coordination_broadcast',
            self.coordination_callback,
            10
        )

        self.status_sub = self.create_subscription(
            String,
            '/status_broadcast',
            self.status_callback,
            10
        )

        # Robot state
        self.current_position = np.array([0.0, 0.0])
        self.current_task = None
        self.teammates = {}  # robot_id -> status info
        self.task_assignments = {}  # task_id -> robot_id

        # Timer for coordination
        self.coordination_timer = self.create_timer(1.0, self.coordination_loop)

        self.get_logger().info(f'Multi-robot coordinator for {self.robot_id} initialized')

    def coordination_callback(self, msg):
        """Handle coordination messages from other robots"""
        try:
            coord_data = json.loads(msg.data)
            sender_id = coord_data['sender_id']

            if coord_data['type'] == 'task_assignment':
                task_id = coord_data['task_id']
                assigned_robot = coord_data['assigned_robot']

                # Update task assignments
                self.task_assignments[task_id] = assigned_robot
                self.get_logger().info(f'Task {task_id} assigned to {assigned_robot}')

            elif coord_data['type'] == 'task_request':
                # Handle request for task assignment
                self.handle_task_request(coord_data)

            elif coord_data['type'] == 'formation_request':
                # Handle formation coordination
                self.handle_formation_request(coord_data)

        except Exception as e:
            self.get_logger().error(f'Error processing coordination message: {e}')

    def status_callback(self, msg):
        """Handle status messages from other robots"""
        try:
            status_data = json.loads(msg.data)
            robot_id = status_data['robot_id']

            # Update teammate information
            self.teammates[robot_id] = {
                'position': status_data['position'],
                'status': status_data['status'],
                'battery': status_data['battery'],
                'timestamp': status_data['timestamp']
            }

            self.get_logger().info(f'Updated status for {robot_id}: {status_data["status"]}')

        except Exception as e:
            self.get_logger().error(f'Error processing status message: {e}')

    def coordination_loop(self):
        """Main coordination loop"""
        # Broadcast own status
        self.broadcast_status()

        # Perform coordination tasks based on current state
        if self.current_task is None:
            # Look for available tasks or request assignment
            self.request_task_assignment()

        # Check formation requirements
        self.maintain_formation()

    def broadcast_status(self):
        """Broadcast current robot status to team"""
        status_msg = {
            'robot_id': self.robot_id,
            'position': self.current_position.tolist(),
            'status': 'active' if self.current_task is None else 'working',
            'battery': 0.8,  # Simulated battery level
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        }

        status_string = String()
        status_string.data = json.dumps(status_msg)
        self.status_pub.publish(status_string)

    def broadcast_coordination(self, coord_type, content):
        """Broadcast coordination message to team"""
        coord_msg = {
            'sender_id': self.robot_id,
            'type': coord_type,
            'content': content,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        }

        coord_string = String()
        coord_string.data = json.dumps(coord_msg)
        self.coordination_pub.publish(coord_string)

    def request_task_assignment(self):
        """Request task assignment from coordinator or other robots"""
        request_content = {
            'capabilities': ['navigation', 'manipulation'],
            'current_position': self.current_position.tolist(),
            'available': True
        }

        self.broadcast_coordination('task_request', request_content)

    def handle_task_request(self, request_data):
        """Handle task request from another robot"""
        requester_id = request_data['sender_id']
        requester_caps = request_data['content']['capabilities']

        # In a real system, this would check for a task coordinator
        # For now, simulate task assignment
        if self.robot_id.startswith('robot_001'):  # Robot 1 acts as coordinator
            # Assign a task to the requesting robot
            task_id = f'task_{int(time.time()) % 1000}'
            task_location = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]

            assignment = {
                'task_id': task_id,
                'location': task_location,
                'assigned_robot': requester_id
            }

            self.broadcast_coordination('task_assignment', assignment)
            self.get_logger().info(f'Assigned task {task_id} to {requester_id}')

    def handle_formation_request(self, request_data):
        """Handle formation request from another robot"""
        formation_type = request_data['content']['formation_type']
        leader_id = request_data['content']['leader']

        if leader_id == self.robot_id:
            # I'm the leader, coordinate formation
            self.coordinate_formation(formation_type)
        elif self.current_task is None:
            # Join the formation as follower
            self.join_formation(formation_type, leader_id)

    def coordinate_formation(self, formation_type):
        """Coordinate formation as leader"""
        if formation_type == 'line':
            # Calculate positions for line formation
            positions = self.calculate_line_formation()
        elif formation_type == 'circle':
            positions = self.calculate_circle_formation()
        else:
            return

        # Assign positions to team members
        for i, (robot_id, _) in enumerate(self.teammates.items()):
            if i < len(positions):
                assignment = {
                    'formation_type': formation_type,
                    'robot_id': robot_id,
                    'target_position': positions[i]
                }
                self.broadcast_coordination('formation_assignment', assignment)

    def join_formation(self, formation_type, leader_id):
        """Join formation as follower"""
        self.get_logger().info(f'Joining {formation_type} formation led by {leader_id}')

    def calculate_line_formation(self):
        """Calculate positions for line formation"""
        positions = []
        spacing = 2.0
        leader_pos = self.current_position  # In real system, get leader position

        for i in range(len(self.teammates) + 1):  # +1 for self
            pos = [leader_pos[0] + i * spacing, leader_pos[1]]
            positions.append(pos)

        return positions

    def calculate_circle_formation(self):
        """Calculate positions for circle formation"""
        positions = []
        radius = 3.0
        num_robots = len(self.teammates) + 1

        for i in range(num_robots):
            angle = 2 * np.pi * i / num_robots
            pos = [
                self.current_position[0] + radius * np.cos(angle),
                self.current_position[1] + radius * np.sin(angle)
            ]
            positions.append(pos)

        return positions

    def maintain_formation(self):
        """Maintain current formation if in one"""
        # In a real system, this would move robots to maintain formation
        pass


def main(args=None):
    rclpy.init(args=args)

    # Create node with specific robot ID
    import sys
    robot_id = sys.argv[1] if len(sys.argv) > 1 else 'robot_001'

    coordinator = MultiRobotCoordinator()
    coordinator.get_parameter('robot_id').set_value(robot_id)

    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Swarm Intelligence Algorithms

### Ant Colony Optimization for Multi-Robot Path Planning

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random


class AntColonyRobot:
    def __init__(self, robot_id, start_pos, target_pos, world_size):
        self.robot_id = robot_id
        self.position = np.array(start_pos, dtype=float)
        self.target = np.array(target_pos, dtype=float)
        self.world_size = world_size
        self.path = [self.position.copy()]
        self.found_target = False
        self.pheromone_trail = []  # Store pheromone trail
        self.carrying_pheromone = True

    def move(self, pheromone_grid, obstacles, evaporation_rate=0.1):
        """Move robot based on pheromone trails and target direction"""
        if self.found_target:
            return  # Robot has reached target

        # Evaporate some pheromone
        if random.random() < evaporation_rate:
            if self.pheromone_trail:
                self.pheromone_trail.pop(0)  # Remove oldest pheromone

        # Calculate possible moves (8 directions)
        possible_moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip staying in place

                new_pos = self.position + np.array([dx, dy])

                # Check bounds
                if (0 <= new_pos[0] < self.world_size and
                    0 <= new_pos[1] < self.world_size):

                    # Check obstacles
                    if not any(np.array_equal(new_pos, obs) for obs in obstacles):
                        possible_moves.append(new_pos)

        if not possible_moves:
            return  # No valid moves

        # Calculate probabilities based on pheromone and distance to target
        probabilities = []
        for move in possible_moves:
            # Pheromone factor (higher pheromone = higher probability)
            pheromone_level = pheromone_grid[int(move[0]), int(move[1])] if 0 <= int(move[0]) < self.world_size and 0 <= int(move[1]) < self.world_size else 0

            # Distance factor (closer to target = higher probability)
            distance_to_target = np.linalg.norm(move - self.target)
            distance_factor = 1.0 / (distance_to_target + 1)  # Closer = higher

            probability = pheromone_level + distance_factor
            probabilities.append(probability)

        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            # If no pheromone, move randomly toward target
            probabilities = [1.0 / len(possible_moves)] * len(possible_moves)

        # Choose move based on probabilities
        chosen_move = possible_moves[np.random.choice(len(possible_moves), p=probabilities)]

        # Update position
        self.position = chosen_move
        self.path.append(self.position.copy())

        # Deposit pheromone
        if 0 <= int(self.position[0]) < self.world_size and 0 <= int(self.position[1]) < self.world_size:
            pheromone_grid[int(self.position[0]), int(self.position[1])] += 0.5

        # Check if reached target
        if np.linalg.norm(self.position - self.target) < 1.0:
            self.found_target = True

    def reset(self, start_pos):
        """Reset robot to new start position"""
        self.position = np.array(start_pos, dtype=float)
        self.path = [self.position.copy()]
        self.found_target = False


class AntColonyMultiRobotSystem:
    def __init__(self, world_size=20, num_robots=5):
        self.world_size = world_size
        self.num_robots = num_robots
        self.pheromone_grid = np.zeros((world_size, world_size))
        self.obstacles = []
        self.robots = []

        # Initialize robots
        for i in range(num_robots):
            start_pos = [random.randint(0, world_size//4), random.randint(0, world_size//4)]
            target_pos = [random.randint(3*world_size//4, world_size-1), random.randint(3*world_size//4, world_size-1)]

            robot = AntColonyRobot(f'robot_{i:02d}', start_pos, target_pos, world_size)
            self.robots.append(robot)

    def add_obstacle(self, pos):
        """Add obstacle to environment"""
        self.obstacles.append(np.array(pos))

    def update(self):
        """Update all robots"""
        for robot in self.robots:
            if not robot.found_target:
                robot.move(self.pheromone_grid, self.obstacles)

    def get_positions(self):
        """Get all robot positions"""
        return [robot.position for robot in self.robots]

    def get_targets(self):
        """Get all target positions"""
        return [robot.target for robot in self.robots]

    def animate_system(self, steps=200):
        """Animate the ant colony system"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect('equal')
        ax.set_title('Ant Colony Optimization for Multi-Robot Path Planning')

        # Initialize scatter plots
        robot_positions = np.array(self.get_positions())
        target_positions = np.array(self.get_targets())

        robot_scatter = ax.scatter(robot_positions[:, 0], robot_positions[:, 1],
                                  c='blue', s=50, label='Robots')
        target_scatter = ax.scatter(target_positions[:, 0], target_positions[:, 1],
                                   c='red', s=100, marker='*', label='Targets')

        # Plot obstacles
        if self.obstacles:
            obstacle_positions = np.array(self.obstacles)
            ax.scatter(obstacle_positions[:, 0], obstacle_positions[:, 1],
                      c='black', s=80, marker='s', label='Obstacles')

        # Plot pheromone heatmap
        im = ax.imshow(self.pheromone_grid, cmap='viridis', alpha=0.3,
                      extent=[0, self.world_size, 0, self.world_size],
                      origin='lower', vmin=0, vmax=1)

        def update(frame):
            self.update()

            # Update robot positions
            robot_positions = np.array(self.get_positions())
            robot_scatter.set_offsets(robot_positions)

            # Update pheromone visualization
            im.set_array(self.pheromone_grid)

            # Update title
            completed = sum(1 for robot in self.robots if robot.found_target)
            ax.set_title(f'Ant Colony Optimization - Step {frame}, Completed: {completed}/{self.num_robots}')

            return [robot_scatter, target_scatter, im]

        anim = FuncAnimation(fig, update, frames=steps, interval=100, blit=False, repeat=True)
        plt.legend()
        plt.show()
        return anim


def main():
    print("Starting Ant Colony Optimization for Multi-Robot Path Planning...")

    # Create system
    ac_system = AntColonyMultiRobotSystem(world_size=20, num_robots=5)

    # Add some obstacles
    for _ in range(10):
        ac_system.add_obstacle([random.randint(5, 15), random.randint(5, 15)])

    print(f"Created system with {ac_system.num_robots} robots and {len(ac_system.obstacles)} obstacles")

    # Run animation
    animation = ac_system.animate_system(steps=300)

    # Count completed robots
    completed = sum(1 for robot in ac_system.robots if robot.found_target)
    print(f"Simulation completed. {completed}/{len(ac_system.robots)} robots reached their targets.")

    return ac_system, animation


if __name__ == '__main__':
    system, anim = main()
```

## Hands-On Lab: Multi-Robot Formation Control

### Objective
Implement a multi-robot formation control system that demonstrates coordination and communication between robots.

### Prerequisites
- Completed Chapter 1-12
- ROS 2 Humble with Gazebo
- Basic understanding of multi-robot systems

### Steps

1. **Create a multi-robot lab package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python multi_robot_lab --dependencies rclpy std_msgs geometry_msgs sensor_msgs nav_msgs tf2_ros
   ```

2. **Create the multi-robot formation controller** (`multi_robot_lab/multi_robot_lab/formation_controller.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import Twist, Pose, Point
   from nav_msgs.msg import Odometry
   from std_msgs.msg import String
   import numpy as np
   import math


   class FormationController(Node):
       def __init__(self):
           super().__init__('formation_controller')

           # Robot ID parameter
           self.declare_parameter('robot_id', 'robot_001')
           self.robot_id = self.get_parameter('robot_id').value

           # Formation parameters
           self.declare_parameter('formation_type', 'line')
           self.declare_parameter('formation_spacing', 2.0)
           self.declare_parameter('leader_id', 'robot_001')

           self.formation_type = self.get_parameter('formation_type').value
           self.formation_spacing = self.get_parameter('formation_spacing').value
           self.leader_id = self.get_parameter('leader_id').value

           # Publishers and subscribers
           self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.robot_id}/cmd_vel', 10)
           self.status_pub = self.create_publisher(String, f'/{self.robot_id}/formation_status', 10)

           # Robot state
           self.current_position = np.array([0.0, 0.0])
           self.current_velocity = np.array([0.0, 0.0])
           self.target_position = np.array([0.0, 0.0])
           self.robot_positions = {}  # robot_id -> position
           self.is_leader = (self.robot_id == self.leader_id)

           # Timer for control loop
           self.control_timer = self.create_timer(0.1, self.control_loop)

           # Subscribe to all robot positions
           for i in range(1, 6):  # Assume up to 5 robots
               robot_name = f'robot_{i:03d}'
               sub = self.create_subscription(
                   Odometry,
                   f'/{robot_name}/odom',
                   lambda msg, rn=robot_name: self.odom_callback(msg, rn),
                   10
               )

           self.get_logger().info(f'Formation controller for {self.robot_id} initialized')

       def odom_callback(self, msg, robot_name):
           """Update robot position from odometry"""
           position = np.array([
               msg.pose.pose.position.x,
               msg.pose.pose.position.y
           ])

           self.robot_positions[robot_name] = position

           # Update own position if this is self
           if robot_name == self.robot_id:
               self.current_position = position

       def calculate_formation_position(self):
           """Calculate target position in formation"""
           if self.is_leader:
               # Leader follows a predefined path or user commands
               # For this example, leader moves in a circle
               time_now = self.get_clock().now().nanoseconds / 1e9
               radius = 5.0
               angle = time_now * 0.5  # Rotate at 0.5 rad/s
               target_x = radius * math.cos(angle)
               target_y = radius * math.sin(angle)
               return np.array([target_x, target_y])

           # Follower robots calculate their position based on leader and formation type
           if self.leader_id not in self.robot_positions:
               return self.current_position  # Stay in place if no leader info

           leader_pos = self.robot_positions[self.leader_id]

           # Calculate robot index in formation
           robot_index = self.get_robot_index(self.robot_id)
           if robot_index == -1:
               return self.current_position

           if self.formation_type == 'line':
               # Line formation: robots arranged in a line behind leader
               # Calculate direction from leader to first follower
               direction = np.array([1.0, 0.0])  # Default direction (could be based on leader's heading)
               offset = self.formation_spacing * robot_index * direction
               target_pos = leader_pos - offset

           elif self.formation_type == 'circle':
               # Circle formation: robots arranged in circle around leader
               angle_between_robots = 2 * math.pi / (self.get_total_robots() - 1)  # -1 to exclude leader
               robot_angle = robot_index * angle_between_robots
               offset_x = self.formation_spacing * math.cos(robot_angle)
               offset_y = self.formation_spacing * math.sin(robot_angle)
               target_pos = leader_pos + np.array([offset_x, offset_y])

           elif self.formation_type == 'diamond':
               # Diamond formation: specific positions relative to leader
               positions = [
                   np.array([0.0, 0.0]),      # Leader position
                   np.array([2.0, 0.0]),      # Right
                   np.array([-2.0, 0.0]),     # Left
                   np.array([0.0, 2.0]),      # Front
                   np.array([0.0, -2.0])      # Back
               ]
               target_pos = leader_pos + positions[robot_index]

           else:
               # Default to line formation
               direction = np.array([1.0, 0.0])
               offset = self.formation_spacing * robot_index * direction
               target_pos = leader_pos - offset

           return target_pos

       def get_robot_index(self, robot_id):
           """Get index of robot in formation list"""
           # Extract number from robot ID (e.g., robot_002 -> 2)
           try:
               robot_num = int(robot_id.split('_')[1])
               return robot_num - 1  # 0-indexed
           except:
               return -1

       def get_total_robots(self):
           """Get total number of robots in system"""
           return len(self.robot_positions)

       def calculate_control_command(self, target_pos):
           """Calculate velocity command to reach target position"""
           error = target_pos - self.current_position
           distance = np.linalg.norm(error)

           cmd_vel = Twist()

           if distance > 0.2:  # If not close to target
               # Proportional controller
               kp_pos = 0.5
               desired_velocity = error * kp_pos

               # Limit velocity
               speed = np.linalg.norm(desired_velocity)
               if speed > 1.0:
                   desired_velocity = desired_velocity / speed

               # Calculate angular velocity to face direction of movement
               if speed > 0.01:
                   desired_angle = math.atan2(desired_velocity[1], desired_velocity[0])
                   current_angle = self.get_current_heading()
                   angle_error = desired_angle - current_angle
                   # Normalize angle error to [-pi, pi]
                   while angle_error > math.pi:
                       angle_error -= 2 * math.pi
                   while angle_error < -math.pi:
                       angle_error += 2 * math.pi

                   cmd_vel.linear.x = min(speed, 1.0)
                   cmd_vel.angular.z = angle_error * 2.0  # Proportional angular control
               else:
                   cmd_vel.linear.x = 0.0
                   cmd_vel.angular.z = 0.0
           else:
               cmd_vel.linear.x = 0.0
               cmd_vel.angular.z = 0.0

           return cmd_vel

       def get_current_heading(self):
           """Get current robot heading from odometry (simplified)"""
           # In a real system, you'd extract this from the orientation quaternion
           # For this example, we'll use a simplified approach
           return 0.0  # Default heading

       def control_loop(self):
           """Main control loop"""
           if len(self.robot_positions) < 2:
               # Not enough robots to form formation
               cmd_vel = Twist()
               self.cmd_vel_pub.publish(cmd_vel)
               return

           # Calculate target position in formation
           self.target_position = self.calculate_formation_position()

           # Calculate control command
           cmd_vel = self.calculate_control_command(self.target_position)

           # Publish command
           self.cmd_vel_pub.publish(cmd_vel)

           # Publish status
           status_msg = String()
           status_msg.data = f"Target: ({self.target_position[0]:.2f}, {self.target_position[1]:.2f}), " \
                            f"Current: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f})"
           self.status_pub.publish(status_msg)

           self.get_logger().info(f'Robot {self.robot_id}: Moving to {self.target_position}, Command: ({cmd_vel.linear.x:.2f}, {cmd_vel.angular.z:.2f})')


   def main(args=None):
       rclpy.init(args=args)

       # Get robot ID from command line or use default
       import sys
       robot_id = sys.argv[1] if len(sys.argv) > 1 else 'robot_001'

       formation_controller = FormationController()
       formation_controller.get_parameter('robot_id').set_value(robot_id)

       try:
           rclpy.spin(formation_controller)
       except KeyboardInterrupt:
           pass
       finally:
           formation_controller.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`multi_robot_lab/launch/formation_demo.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory
   import os


   def generate_launch_description():
       # Declare launch arguments
       formation_type = DeclareLaunchArgument(
           'formation_type',
           default_value='line',
           description='Formation type: line, circle, or diamond'
       )

       formation_spacing = DeclareLaunchArgument(
           'formation_spacing',
           default_value='2.0',
           description='Spacing between robots in formation'
       )

       # Launch multiple robot controllers
       robot_controllers = []
       for i in range(1, 6):  # Launch 5 robots
           robot_id = f'robot_{i:03d}'
           controller_node = Node(
               package='multi_robot_lab',
               executable='formation_controller',
               name=f'formation_controller_{robot_id}',
               parameters=[
                   {'robot_id': robot_id},
                   {'formation_type': LaunchConfiguration('formation_type')},
                   {'formation_spacing': LaunchConfiguration('formation_spacing')},
                   {'leader_id': 'robot_001'}  # Robot 1 is leader
               ],
               output='screen'
           )
           robot_controllers.append(controller_node)

       return LaunchDescription([
           formation_type,
           formation_spacing,
       ] + robot_controllers)
   ```

4. **Update setup.py**:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'multi_robot_lab'

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
       description='Multi-robot lab for formations and coordination',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'formation_controller = multi_robot_lab.formation_controller:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select multi_robot_lab
   source install/setup.bash
   ```

6. **Run the multi-robot formation simulation**:
   ```bash
   ros2 launch multi_robot_lab formation_demo.launch.py formation_type:=circle formation_spacing:=3.0
   ```

### Expected Results
- Multiple robots should form the specified formation (line, circle, or diamond)
- Robots should maintain formation while the leader moves
- Followers should adjust their positions based on leader movement
- Formation should be stable and robust to minor disturbances

### Troubleshooting Tips
- Ensure all robot namespaces are correctly set up
- Check that odometry topics are being published for each robot
- Verify TF frames are properly configured for each robot
- Monitor the logs for formation status and control commands

## Summary

In this chapter, we've explored the fundamental concepts of multi-robot systems and coordination, including:

1. **Multi-Robot Architectures**: Centralized vs decentralized approaches
2. **Communication Protocols**: Robot-to-robot communication and message passing
3. **Swarm Robotics**: Collective behavior algorithms and emergent coordination
4. **Task Allocation**: Auction-based and market-based task assignment
5. **Coordination Algorithms**: Consensus, formation control, and distributed decision making
6. **Implementation**: Practical examples of multi-robot coordination in ROS 2

The hands-on lab provided experience with creating a formation control system that demonstrates coordination between multiple robots. This foundation is essential for more advanced multi-robot applications including cooperative manipulation, distributed sensing, and collective decision making that we'll explore in the upcoming chapters.