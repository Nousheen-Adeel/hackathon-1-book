---
title: Chapter 10 - Reinforcement Learning for Robotics
sidebar_position: 1
---

# Chapter 10: Reinforcement Learning for Robotics

## Learning Goals

- Apply RL algorithms to robotic control
- Understand simulation-to-reality transfer
- Learn policy optimization techniques
- Train RL agents for basic robotic tasks
- Transfer policies from simulation to real robots
- Optimize control policies

## Introduction to Reinforcement Learning for Robotics

Reinforcement Learning (RL) is a powerful machine learning paradigm where agents learn to make decisions by interacting with an environment to maximize cumulative rewards. In robotics, RL has emerged as a transformative approach for learning complex behaviors and control policies that are difficult to program explicitly.

### Key Concepts in RL

- **Agent**: The learning entity (the robot)
- **Environment**: The world the agent interacts with
- **State**: The current situation of the agent
- **Action**: What the agent can do
- **Reward**: Feedback signal for the agent's actions
- **Policy**: Strategy that maps states to actions
- **Value Function**: Expected future rewards from a state

### Why RL for Robotics?

Traditional control methods require explicit mathematical models and manual tuning. RL offers several advantages:

1. **Adaptability**: Learns to handle uncertainties and disturbances
2. **Optimization**: Finds optimal behaviors without manual tuning
3. **Complex Tasks**: Handles high-dimensional state and action spaces
4. **Generalization**: Can adapt to new situations within the learned policy

## Markov Decision Processes (MDPs)

The foundation of RL is the Markov Decision Process, which assumes that the future state depends only on the current state and action, not on the history of previous states.

### MDP Components

```python
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt


class RobotMDP:
    def __init__(self, state_space, action_space, reward_function, transition_function):
        """
        Initialize a Markov Decision Process for robotics
        state_space: Space of possible states
        action_space: Space of possible actions
        reward_function: Function R(s, a) -> reward
        transition_function: Function T(s, a) -> probability distribution over next states
        """
        self.state_space = state_space
        self.action_space = action_space
        self.reward_function = reward_function
        self.transition_function = transition_function

        # Current state of the robot
        self.current_state = None

    def reset(self):
        """Reset the environment to initial state"""
        self.current_state = self.state_space.sample()
        return self.current_state

    def step(self, action):
        """
        Execute an action and return (next_state, reward, done, info)
        """
        if self.current_state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # Get reward for current state-action pair
        reward = self.reward_function(self.current_state, action)

        # Get next state from transition function
        next_state = self.transition_function(self.current_state, action)

        # Update current state
        self.current_state = next_state

        # For now, assume episode doesn't end
        done = False
        info = {}

        return next_state, reward, done, info


# Example: Simple navigation MDP
class NavigationMDP(RobotMDP):
    def __init__(self, grid_size=10, goal_position=(9, 9), obstacles=None):
        """
        Simple grid-world navigation MDP
        grid_size: Size of the square grid
        goal_position: Position of the goal (row, col)
        obstacles: List of obstacle positions
        """
        self.grid_size = grid_size
        self.goal_position = goal_position
        self.obstacles = obstacles if obstacles else []

        # State space: (x, y) positions
        self.state_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([grid_size-1, grid_size-1]),
            dtype=np.int32
        )

        # Action space: up, down, left, right
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right

        # Reward function
        def reward_function(state, action):
            next_state = self._get_next_state(state, action)

            # Check if next state is obstacle
            if tuple(next_state) in self.obstacles:
                return -10  # Large penalty for hitting obstacle

            # Distance-based reward
            distance_to_goal = np.linalg.norm(np.array(next_state) - np.array(self.goal_position))
            reward = -distance_to_goal  # Negative distance as reward

            # Bonus for reaching goal
            if np.array_equal(next_state, self.goal_position):
                reward += 100

            return reward

        # Transition function
        def transition_function(state, action):
            return self._get_next_state(state, action)

        super().__init__(self.state_space, self.action_space, reward_function, transition_function)

    def _get_next_state(self, state, action):
        """Get next state based on action (with some stochasticity)"""
        x, y = state
        new_x, new_y = x, y

        # With 10% probability, action fails (stochastic transitions)
        if np.random.random() < 0.1:
            # Stay in same position
            return np.array([x, y])

        # Execute action
        if action == 0:  # Up
            new_y = max(0, y - 1)
        elif action == 1:  # Down
            new_y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # Left
            new_x = max(0, x - 1)
        elif action == 3:  # Right
            new_x = min(self.grid_size - 1, x + 1)

        # Check if new position is obstacle
        if (new_x, new_y) in self.obstacles:
            return np.array([x, y])  # Stay in place if obstacle

        return np.array([new_x, new_y])

    def reset(self):
        """Reset to random non-goal, non-obstacle position"""
        while True:
            state = self.state_space.sample()
            if not (tuple(state) == self.goal_position or tuple(state) in self.obstacles):
                self.current_state = state
                return state


# Example usage
def main():
    # Create navigation environment
    env = NavigationMDP(grid_size=10, goal_position=(8, 8), obstacles=[(3, 3), (3, 4), (3, 5)])

    # Reset environment
    state = env.reset()
    print(f"Initial state: {state}")

    # Run a few steps
    total_reward = 0
    for step in range(20):
        # Random action
        action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)

        print(f"Step {step}: Action={action}, State={state} -> {next_state}, Reward={reward:.2f}")

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Total reward: {total_reward:.2f}")


if __name__ == '__main__':
    main()
```

## Value-Based Methods

### Q-Learning

Q-Learning is a model-free RL algorithm that learns the value of state-action pairs:

```python
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initialize Q-Learning agent
        action_space: Number of possible actions
        learning_rate: Learning rate (alpha)
        discount_factor: Discount factor (gamma)
        epsilon: Exploration rate for epsilon-greedy
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Q-table: state -> [Q-value for each action]
        self.q_table = defaultdict(lambda: np.zeros(action_space))

    def get_action(self, state, training=True):
        """
        Get action using epsilon-greedy policy
        state: Current state
        training: If True, use exploration; if False, use greedy policy
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_space - 1)
        else:
            # Exploit: best action according to Q-table
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Bellman equation
        """
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

        # Update Q-value
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)


# Deep Q-Network (DQN) implementation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Deep Q-Network agent
        state_dim: Dimension of state space
        action_dim: Number of possible actions
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Main network
        self.q_network = DQN(state_dim, 128, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Target network (for stable training)
        self.target_network = DQN(state_dim, 128, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Replay buffer
        self.memory = []
        self.max_memory_size = 10000

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory_size:
            del self.memory[0]

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train on batch of experiences from memory"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())


# Example: Training DQN on a simple environment
def train_dqn_example():
    """Example of training DQN on CartPole environment"""
    import gym

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    episodes = 500
    scores = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle newer gym versions
            state = state[0]
        total_reward = 0

        for step in range(200):  # Max steps per episode
            action = agent.act(state)
            result = env.step(action)

            # Handle different return formats
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

        # Train the agent
        if len(agent.memory) > 32:
            agent.replay(32)

        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Scores over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.subplot(1, 2, 2)
    avg_scores = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plt.plot(avg_scores)
    plt.title('Average Scores (100-episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')

    plt.tight_layout()
    plt.show()

    return agent


if __name__ == '__main__':
    # Run the example
    trained_agent = train_dqn_example()
```

## Policy Gradient Methods

Policy gradient methods directly optimize the policy function:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr_policy=1e-3, lr_value=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)

        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr_value)

        self.states = []
        self.actions = []
        self.rewards = []

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def put_data(self, transition):
        self.states.append(transition[0])
        self.actions.append(transition[1])
        self.rewards.append(transition[2])

    def train_net(self):
        if len(self.rewards) == 0:
            return

        # Calculate discounted rewards
        R = 0
        discounted_rewards = []
        for reward in self.rewards[::-1]:
            R = reward + 0.99 * R
            discounted_rewards.insert(0, R)

        # Normalize discounted rewards
        discounted_rewards = torch.tensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)

        # Calculate policy loss
        probs = self.policy_net(states)
        m = Categorical(probs)
        log_probs = m.log_prob(actions)

        policy_loss = -(log_probs * discounted_rewards.detach()).mean()

        # Update policy network
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # Reset buffers
        self.states, self.actions, self.rewards = [], [], []


# Actor-Critic implementation
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor layers
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)

        # Critic layers
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        # Actor
        actor_x = F.relu(self.actor_fc(x))
        action_probs = F.softmax(self.actor_out(actor_x), dim=-1)

        # Critic
        critic_x = F.relu(self.critic_fc(x))
        state_value = self.critic_out(critic_x)

        return action_probs, state_value


class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, state_value = self.model(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()

        return action.item(), action_dist.log_prob(action), state_value


def train_reinforce_example():
    """Example of training REINFORCE on CartPole"""
    import gym

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim)

    episodes = 1000
    scores = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle newer gym versions
            state = state[0]
        total_reward = 0

        while True:
            action, log_prob = agent.act(state)
            result = env.step(action)

            # Handle different return formats
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            agent.put_data((state, action, reward))

            state = next_state
            total_reward += reward

            if done:
                break

        agent.train_net()
        scores.append(total_reward)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")

    return agent


def train_a2c_example():
    """Example of training A2C on CartPole"""
    import gym

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim)

    episodes = 1000
    scores = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle newer gym versions
            state = state[0]
        total_reward = 0

        log_probs = []
        values = []
        rewards = []

        for step in range(200):  # Max steps per episode
            action, log_prob, value = agent.act(state)
            result = env.step(action)

            # Handle different return formats
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

        # Calculate returns and advantages
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + agent.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).float().to(agent.device)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()

        # Calculate advantage
        advantages = returns - values

        # Calculate losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        # Update networks
        agent.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        agent.optimizer.step()

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")

    return agent


if __name__ == '__main__':
    print("Training REINFORCE agent...")
    reinforce_agent = train_reinforce_example()

    print("\nTraining A2C agent...")
    a2c_agent = train_a2c_example()
```

## Deep Deterministic Policy Gradient (DDPG)

DDPG is particularly useful for continuous action spaces common in robotics:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.relu(self.l1(sa))
        q = torch.relu(self.l2(q))
        q = self.l3(q)
        return q


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005, batch_size=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma  # Discount factor
        self.tau = tau      # Target network update rate
        self.batch_size = batch_size

        # Replay buffer
        self.replay_buffer = deque(maxlen=100000)

        # Noise for exploration
        self.noise = np.zeros(action_dim)

    def select_action(self, state, add_noise=True, noise_scale=0.1):
        """Select action with optional noise for exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            # Add Ornstein-Uhlenbeck noise for exploration
            self.noise = self.noise * 0.9 + np.random.normal(0, noise_scale, size=len(self.noise))
            action += self.noise

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        """Update networks using batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.BoolTensor(done).unsqueeze(1).to(self.device)

        # Compute target Q-values
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action)
        target_Q = reward + (self.gamma * target_Q * ~done)

        # Critic update
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def train_ddpg_example():
    """Example of training DDPG on a continuous control environment"""
    import gym

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)

    episodes = 1000
    scores = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle newer gym versions
            state = state[0]
        total_reward = 0

        for step in range(200):  # Max steps per episode
            action = agent.select_action(state)
            result = env.step(action)

            # Handle different return formats
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")

    return agent


if __name__ == '__main__':
    print("Training DDPG agent...")
    ddpg_agent = train_ddpg_example()
```

## Robotics-Specific RL Applications

### Robot Control with RL

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class RobotArmEnv:
    def __init__(self, arm_length=1.0, target=(0.5, 0.5)):
        """
        Simple 2D robot arm environment
        arm_length: Length of each arm segment
        target: Target position (x, y)
        """
        self.arm_length = arm_length
        self.target = np.array(target)
        self.state_dim = 4  # [x, y, theta1, theta2] (end effector position and joint angles)
        self.action_dim = 2  # [delta_theta1, delta_theta2] (change in joint angles)

        # Action limits (radians)
        self.action_limits = np.array([-0.1, 0.1])  # Small changes for stability

        self.reset()

    def reset(self):
        """Reset environment to random initial state"""
        # Random initial joint angles
        self.joint_angles = np.random.uniform(-np.pi, np.pi, 2)
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        """Get current state [x, y, theta1, theta2]"""
        # Calculate end effector position from joint angles
        x = self.arm_length * np.cos(self.joint_angles[0]) + self.arm_length * np.cos(self.joint_angles[0] + self.joint_angles[1])
        y = self.arm_length * np.sin(self.joint_angles[0]) + self.arm_length * np.sin(self.joint_angles[0] + self.joint_angles[1])

        return np.array([x, y, self.joint_angles[0], self.joint_angles[1]])

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Apply action (change joint angles)
        self.joint_angles += action

        # Clamp joint angles to reasonable range
        self.joint_angles = np.clip(self.joint_angles, -np.pi, np.pi)

        # Get new state
        self.state = self._get_state()

        # Calculate reward (negative distance to target + bonus for being close)
        distance_to_target = np.linalg.norm(self.state[:2] - self.target)
        reward = -distance_to_target

        # Bonus for getting very close to target
        if distance_to_target < 0.1:
            reward += 10

        done = False  # Never done for this example

        return self.state, reward, done, {}

    def render(self):
        """Visualize the robot arm"""
        # Calculate arm segment endpoints
        x1 = self.arm_length * np.cos(self.joint_angles[0])
        y1 = self.arm_length * np.sin(self.joint_angles[0])

        x2 = x1 + self.arm_length * np.cos(self.joint_angles[0] + self.joint_angles[1])
        y2 = y1 + self.arm_length * np.sin(self.joint_angles[0] + self.joint_angles[1])

        plt.figure(figsize=(8, 8))
        plt.plot([0, x1, x2], [0, y1, y2], 'o-', linewidth=3, markersize=8, label='Robot Arm')
        plt.plot(self.target[0], self.target[1], 'r*', markersize=15, label='Target')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title('Robot Arm Environment')
        plt.axis('equal')
        plt.show()


class RobotArmDDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64

        self.replay_buffer = deque(maxlen=10000)
        self.noise_scale = 0.1

    def select_action(self, state, add_noise=True):
        """Select action with optional noise for exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action += noise

        return np.clip(action, -1.0, 1.0)  # Clip to valid range

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        """Update networks using batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.BoolTensor(done).unsqueeze(1).to(self.device)

        # Compute target Q-values
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action)
        target_Q = reward + (self.gamma * target_Q * ~done)

        # Critic update
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def train_robot_arm():
    """Train RL agent to control robot arm to reach target"""
    env = RobotArmEnv(target=(1.0, 0.5))
    agent = RobotArmDDPGAgent(env.state_dim, env.action_dim)

    episodes = 500
    scores = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(100):  # Max 100 steps per episode
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")

    # Test the trained agent
    print("\nTesting trained agent...")
    state = env.reset()
    trajectory = [env.state[:2].copy()]  # Store end effector positions

    for step in range(100):
        action = agent.select_action(state, add_noise=False)  # No noise during testing
        state, reward, done, _ = env.step(action)
        trajectory.append(state[:2].copy())

        if np.linalg.norm(state[:2] - env.target) < 0.1:  # Close enough to target
            print(f"Target reached in {step+1} steps!")
            break

    # Plot trajectory
    trajectory = np.array(trajectory)
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.plot(env.target[0], env.target[1], 'r*', markersize=15, label='Target')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    plt.title('Robot Arm Trajectory After Training')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    return agent


if __name__ == '__main__':
    trained_agent = train_robot_arm()
```

## Simulation-to-Reality Transfer

### Domain Randomization

Domain randomization helps bridge the sim-to-real gap:

```python
import torch
import numpy as np


class DomainRandomizationEnv:
    def __init__(self, base_env, randomization_params=None):
        """
        Wrap an environment with domain randomization
        base_env: The original environment
        randomization_params: Dict of parameters to randomize with their ranges
        """
        self.base_env = base_env
        self.randomization_params = randomization_params or {}

        # Store original parameters
        self.original_params = {}
        for param_name, (min_val, max_val) in self.randomization_params.items():
            if hasattr(base_env, param_name):
                self.original_params[param_name] = getattr(base_env, param_name)

    def reset(self):
        """Reset environment with randomized parameters"""
        # Randomize parameters
        for param_name, (min_val, max_val) in self.randomization_params.items():
            if hasattr(self.base_env, param_name):
                random_value = np.random.uniform(min_val, max_val)
                setattr(self.base_env, param_name, random_value)

        return self.base_env.reset()

    def step(self, action):
        """Take step in randomized environment"""
        return self.base_env.step(action)

    def render(self):
        """Render the environment"""
        return self.base_env.render()


# Example: Randomizing robot dynamics parameters
class RandomizedRobotEnv:
    def __init__(self, robot_mass_range=(0.8, 1.2), friction_range=(0.1, 0.3)):
        self.robot_mass_range = robot_mass_range
        self.friction_range = friction_range

        # Original parameters
        self.original_mass = 1.0
        self.original_friction = 0.2

        # Current randomized parameters
        self.robot_mass = self.original_mass
        self.friction = self.original_friction

    def reset(self):
        """Randomize parameters and reset"""
        self.robot_mass = np.random.uniform(*self.robot_mass_range)
        self.friction = np.random.uniform(*self.friction_range)

        # Reset robot state (simplified)
        self.state = np.random.uniform(-1, 1, 4)  # [x, y, vx, vy]
        return self.state

    def step(self, action):
        """Simulate robot dynamics with randomized parameters"""
        # Simplified physics simulation
        dt = 0.01

        # Update velocities based on action and friction
        self.state[2] += (action[0] / self.robot_mass - self.friction * self.state[2]) * dt  # vx
        self.state[3] += (action[1] / self.robot_mass - self.friction * self.state[3]) * dt  # vy

        # Update positions
        self.state[0] += self.state[2] * dt  # x
        self.state[1] += self.state[3] * dt  # y

        # Calculate reward (negative distance to origin)
        distance = np.linalg.norm(self.state[:2])
        reward = -distance

        # Simple termination condition
        done = distance > 10  # Terminate if too far from origin

        return self.state.copy(), reward, done, {}


def train_with_domain_randomization():
    """Example of training with domain randomization"""
    # Create randomized environment
    env = RandomizedRobotEnv(robot_mass_range=(0.7, 1.3), friction_range=(0.05, 0.35))

    # Initialize RL agent
    state_dim = 4  # [x, y, vx, vy]
    action_dim = 2  # [force_x, force_y]
    max_action = 1.0

    agent = RobotArmDDPGAgent(state_dim, action_dim, max_action)

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(100):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
            print(f"  Robot Mass: {env.robot_mass:.2f}, Friction: {env.friction:.2f}")

    return agent


if __name__ == '__main__':
    print("Training with domain randomization...")
    agent = train_with_domain_randomization()
```

## ROS 2 Integration for RL

### RL Node for Robotics

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, JointState
from nav_msgs.msg import Odometry
import numpy as np
import torch
import torch.nn as nn


class RobotRLAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RobotRLAgent, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Actions between -1 and 1
        return x


class RLNavigationNode(Node):
    def __init__(self):
        super().__init__('rl_navigation_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rl_status_pub = self.create_publisher(Bool, '/rl_active', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        # RL components
        self.state_dim = 20  # Example: 18 laser readings + 2 position values
        self.action_dim = 2  # linear and angular velocity
        self.rl_agent = RobotRLAgent(self.state_dim, self.action_dim)

        # Robot state
        self.current_pose = None
        self.current_scan = None
        self.current_joints = None
        self.rl_active = False

        # Navigation parameters
        self.linear_vel_limit = 0.5
        self.angular_vel_limit = 1.0

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('RL Navigation node initialized')

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.current_scan = msg

    def joint_callback(self, msg):
        """Update joint state data"""
        self.current_joints = msg

    def get_robot_state(self):
        """Construct state vector from sensor data"""
        if self.current_scan is None or self.current_pose is None:
            # Return a default state if no data
            return np.zeros(self.state_dim)

        # Example state construction:
        # 18 laser readings (every 20th reading from 360-degree scan)
        laser_data = []
        if len(self.current_scan.ranges) >= 18:
            step = len(self.current_scan.ranges) // 18
            for i in range(0, len(self.current_scan.ranges), step):
                if i < len(self.current_scan.ranges):
                    # Normalize laser range (0 to 10m -> 0 to 1)
                    range_val = min(self.current_scan.ranges[i], 10.0) / 10.0
                    laser_data.append(range_val)

        # Pad with zeros if not enough readings
        while len(laser_data) < 18:
            laser_data.append(1.0)  # Max distance = no obstacle

        # 2D position (normalize to reasonable range)
        pos_x = self.current_pose.position.x / 10.0  # Assuming max 10m range
        pos_y = self.current_pose.position.y / 10.0

        # Construct full state vector
        state = np.array(laser_data + [pos_x, pos_y])

        return state

    def control_loop(self):
        """Main control loop for RL-based navigation"""
        if not self.rl_active:
            return

        # Get current state
        state = self.get_robot_state()

        # Get action from RL agent
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = self.rl_agent(state_tensor)
        action = action_tensor.detach().numpy()[0]

        # Convert normalized action to actual velocities
        linear_vel = action[0] * self.linear_vel_limit
        angular_vel = action[1] * self.angular_vel_limit

        # Create and publish velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel

        self.cmd_vel_pub.publish(cmd_vel)

        # Log action
        self.get_logger().info(f'RL Action - Linear: {linear_vel:.3f}, Angular: {angular_vel:.3f}')

    def activate_rl(self):
        """Activate RL navigation"""
        self.rl_active = True
        status_msg = Bool()
        status_msg.data = True
        self.rl_status_pub.publish(status_msg)
        self.get_logger().info('RL navigation activated')

    def deactivate_rl(self):
        """Deactivate RL navigation"""
        self.rl_active = False
        status_msg = Bool()
        status_msg.data = False
        self.rl_status_pub.publish(status_msg)

        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        self.get_logger().info('RL navigation deactivated')


def main(args=None):
    rclpy.init(args=args)
    rl_navigation_node = RLNavigationNode()

    try:
        # Activate RL after a short delay
        def activate_timer_callback():
            rl_navigation_node.activate_rl()
            activate_timer.cancel()

        activate_timer = rl_navigation_node.create_timer(2.0, activate_timer_callback)

        rclpy.spin(rl_navigation_node)
    except KeyboardInterrupt:
        pass
    finally:
        rl_navigation_node.deactivate_rl()
        rl_navigation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Hands-On Lab: RL-Based Robot Control

### Objective
Create a complete reinforcement learning system that trains a robot to perform a navigation task in simulation, then deploy the learned policy to control the robot.

### Prerequisites
- Completed Chapter 1-10
- ROS 2 Humble with Gazebo and Navigation2
- PyTorch installed
- Basic understanding of RL concepts

### Steps

1. **Create an RL lab package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python rl_robot_lab --dependencies rclpy geometry_msgs nav_msgs sensor_msgs std_msgs torch gym numpy matplotlib
   ```

2. **Create the main RL training node** (`rl_robot_lab/rl_robot_lab/rl_training_node.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import Float32MultiArray, Bool
   from geometry_msgs.msg import Twist, Pose
   from sensor_msgs.msg import LaserScan
   from nav_msgs.msg import Odometry
   import numpy as np
   import torch
   import torch.nn as nn
   import torch.optim as optim
   import random
   from collections import deque
   import copy


   class DQN(nn.Module):
       def __init__(self, state_dim, action_dim, hidden_dim=128):
           super(DQN, self).__init__()
           self.fc1 = nn.Linear(state_dim, hidden_dim)
           self.fc2 = nn.Linear(hidden_dim, hidden_dim)
           self.fc3 = nn.Linear(hidden_dim, action_dim)

       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = torch.relu(self.fc2(x))
           x = self.fc3(x)
           return x


   class DQNAgent:
       def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

           self.state_dim = state_dim
           self.action_dim = action_dim
           self.gamma = gamma
           self.epsilon = epsilon
           self.epsilon_decay = epsilon_decay
           self.epsilon_min = epsilon_min

           # Main network
           self.q_network = DQN(state_dim, action_dim).to(self.device)
           self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

           # Target network
           self.target_network = DQN(state_dim, action_dim).to(self.device)
           self.target_network.load_state_dict(self.q_network.state_dict())

           # Replay buffer
           self.memory = deque(maxlen=10000)

       def remember(self, state, action, reward, next_state, done):
           """Store experience in replay buffer"""
           self.memory.append((state, action, reward, next_state, done))

       def act(self, state, training=True):
           """Choose action using epsilon-greedy policy"""
           if training and np.random.random() <= self.epsilon:
               return random.randrange(self.action_dim)

           state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
           q_values = self.q_network(state_tensor)
           return np.argmax(q_values.cpu().data.numpy())

       def replay(self, batch_size=32):
           """Train on batch of experiences from memory"""
           if len(self.memory) < batch_size:
               return

           batch = random.sample(self.memory, batch_size)
           states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
           actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
           rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
           next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
           dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

           current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
           next_q_values = self.target_network(next_states).max(1)[0].detach()
           target_q_values = rewards + (self.gamma * next_q_values * ~dones)

           loss = nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)

           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()

           # Decay epsilon
           if self.epsilon > self.epsilon_min:
               self.epsilon *= self.epsilon_decay

       def update_target_network(self):
           """Copy weights from main network to target network"""
           self.target_network.load_state_dict(self.q_network.state_dict())


   class RLTrainingNode(Node):
       def __init__(self):
           super().__init__('rl_training_node')

           # Publishers
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.training_status_pub = self.create_publisher(Bool, '/training_active', 10)

           # Subscribers
           self.odom_sub = self.create_subscription(
               Odometry,
               '/odom',
               self.odom_callback,
               10
           )

           self.scan_sub = self.create_subscription(
               LaserScan,
               '/scan',
               self.scan_callback,
               10
           )

           # RL components
           self.state_dim = 18 + 2  # 18 laser readings + 2 position values
           self.action_dim = 4  # 4 discrete actions: forward, backward, left, right
           self.rl_agent = DQNAgent(self.state_dim, self.action_dim)

           # Robot state
           self.current_pose = None
           self.current_scan = None
           self.rl_training_active = False
           self.episode_step = 0
           self.max_episode_steps = 200
           self.total_reward = 0.0
           self.episode_count = 0

           # Training parameters
           self.training_batch_size = 32
           self.target_update_freq = 100
           self.step_count = 0

           # Timer for control loop
           self.control_timer = self.create_timer(0.1, self.control_loop)

           self.get_logger().info('RL Training node initialized')

       def odom_callback(self, msg):
           """Update robot pose from odometry"""
           self.current_pose = msg.pose.pose

       def scan_callback(self, msg):
           """Update laser scan data"""
           self.current_scan = msg

       def get_robot_state(self):
           """Construct state vector from sensor data"""
           if self.current_scan is None or self.current_pose is None:
               # Return a default state if no data
               return np.zeros(self.state_dim)

           # 18 laser readings (every 20th reading from 360-degree scan)
           laser_data = []
           if len(self.current_scan.ranges) >= 18:
               step = len(self.current_scan.ranges) // 18
               for i in range(0, len(self.current_scan.ranges), step):
                   if i < len(self.current_scan.ranges):
                       # Normalize laser range (0 to 10m -> 0 to 1)
                       range_val = min(self.current_scan.ranges[i], 10.0) / 10.0
                       laser_data.append(range_val)

           # Pad with zeros if not enough readings
           while len(laser_data) < 18:
               laser_data.append(1.0)  # Max distance = no obstacle

           # 2D position (normalize to reasonable range)
           pos_x = self.current_pose.position.x / 10.0  # Assuming max 10m range
           pos_y = self.current_pose.position.y / 10.0

           # Construct full state vector
           state = np.array(laser_data + [pos_x, pos_y])

           return state

       def calculate_reward(self, state, action):
           """Calculate reward based on current state and action"""
           # Extract laser readings and position
           laser_readings = state[:18]
           pos_x, pos_y = state[18], state[17]

           # Reward for moving forward (away from walls in front)
           min_front_dist = min(laser_readings[7:11])  # Front readings
           forward_reward = min_front_dist * 10  # Higher reward for being away from obstacles

           # Penalty for being too close to obstacles
           obstacle_penalty = 0
           for dist in laser_readings:
               if dist < 0.3:  # Very close to obstacle
                   obstacle_penalty -= 10
               elif dist < 0.6:  # Moderately close
                   obstacle_penalty -= 2

           # Small penalty for each step to encourage efficiency
           time_penalty = -0.1

           # Reward for exploring new areas (simplified)
           exploration_bonus = 0.1 * (abs(pos_x) + abs(pos_y)) / 10.0

           total_reward = forward_reward + obstacle_penalty + time_penalty + exploration_bonus
           return total_reward

       def get_action_description(self, action):
           """Get human-readable description of action"""
           actions = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']
           return actions[action] if 0 <= action < len(actions) else 'UNKNOWN'

       def control_loop(self):
           """Main control loop for RL training"""
           if not self.rl_training_active:
               return

           # Get current state
           current_state = self.get_robot_state()

           # Choose action using RL agent
           action = self.rl_agent.act(current_state, training=True)

           # Calculate reward for previous action
           if self.episode_step > 0:
               reward = self.calculate_reward(current_state, action)
               self.total_reward += reward

               # Store experience in replay buffer
               self.rl_agent.remember(self.previous_state, self.previous_action, reward, current_state, False)

               # Train the agent
               if len(self.rl_agent.memory) > self.training_batch_size:
                   self.rl_agent.replay(self.training_batch_size)

               # Update target network periodically
               self.step_count += 1
               if self.step_count % self.target_update_freq == 0:
                   self.rl_agent.update_target_network()

           # Execute action
           cmd_vel = Twist()
           if action == 0:  # FORWARD
               cmd_vel.linear.x = 0.3
               cmd_vel.angular.z = 0.0
           elif action == 1:  # BACKWARD
               cmd_vel.linear.x = -0.2
               cmd_vel.angular.z = 0.0
           elif action == 2:  # LEFT
               cmd_vel.linear.x = 0.1
               cmd_vel.angular.z = 0.5
           elif action == 3:  # RIGHT
               cmd_vel.linear.x = 0.1
               cmd_vel.angular.z = -0.5

           self.cmd_vel_pub.publish(cmd_vel)

           # Store state and action for next iteration
           self.previous_state = current_state.copy()
           self.previous_action = action

           # Update episode tracking
           self.episode_step += 1

           # Check if episode should end
           if self.episode_step >= self.max_episode_steps:
               self.end_episode()

           # Log information periodically
           if self.step_count % 50 == 0:
               self.get_logger().info(
                   f'Episode {self.episode_count}, Step {self.episode_step}, '
                   f'Epsilon: {self.rl_agent.epsilon:.3f}, '
                   f'Total Reward: {self.total_reward:.2f}, '
                   f'Action: {self.get_action_description(action)}'
               )

       def end_episode(self):
           """End current episode and start new one"""
           self.get_logger().info(
               f'Episode {self.episode_count} ended. Total reward: {self.total_reward:.2f}, '
               f'Steps: {self.episode_step}'
           )

           # Reset episode variables
           self.episode_step = 0
           self.total_reward = 0.0
           self.episode_count += 1

       def start_training(self):
           """Start RL training"""
           self.rl_training_active = True
           status_msg = Bool()
           status_msg.data = True
           self.training_status_pub.publish(status_msg)
           self.get_logger().info('RL training started')

       def stop_training(self):
           """Stop RL training"""
           self.rl_training_active = False
           status_msg = Bool()
           status_msg.data = False
           self.training_status_pub.publish(status_msg)

           # Stop robot
           stop_cmd = Twist()
           self.cmd_vel_pub.publish(stop_cmd)
           self.get_logger().info('RL training stopped')


   def main(args=None):
       rclpy.init(args=args)
       rl_training_node = RLTrainingNode()

       try:
           # Start training after a short delay
           def start_training_timer_callback():
               rl_training_node.start_training()
               start_timer.cancel()

           start_timer = rl_training_node.create_timer(3.0, start_training_timer_callback)

           rclpy.spin(rl_training_node)
       except KeyboardInterrupt:
           pass
       finally:
           rl_training_node.stop_training()
           rl_training_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`rl_robot_lab/launch/rl_training.launch.py`):
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

       # RL training node
       rl_training_node = Node(
           package='rl_robot_lab',
           executable='rl_training_node',
           name='rl_training_node',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
           output='screen'
       )

       return LaunchDescription([
           use_sim_time,
           rl_training_node
       ])
   ```

4. **Update setup.py**:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'rl_robot_lab'

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
       description='RL robot lab for robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'rl_training_node = rl_robot_lab.rl_training_node:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select rl_robot_lab
   source install/setup.bash
   ```

6. **Run the RL training system**:
   ```bash
   ros2 launch rl_robot_lab rl_training.launch.py
   ```

### Expected Results
- The robot should learn to navigate while avoiding obstacles
- The RL agent should improve its policy over time
- Reward values should increase as the agent learns
- The robot should develop effective navigation behaviors

### Troubleshooting Tips
- Ensure PyTorch is properly installed
- Verify that sensor topics are being published correctly
- Monitor the epsilon decay to ensure proper exploration-exploitation balance
- Check that the reward function is providing meaningful feedback

## Summary

In this chapter, we've explored the application of reinforcement learning to robotics, covering fundamental algorithms like Q-Learning, Deep Q-Networks, Actor-Critic methods, and Deep Deterministic Policy Gradient. We've implemented practical examples of each approach and discussed simulation-to-reality transfer techniques like domain randomization.

The hands-on lab provided experience with creating a complete RL system for robot navigation, demonstrating how to integrate RL algorithms with ROS 2 and apply them to real robotic tasks. This foundation is essential for more advanced applications in robotics intelligence and autonomy that we'll explore in the upcoming chapters.