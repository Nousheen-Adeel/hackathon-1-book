---
title: Chapter 11 - Imitation Learning and VLA
sidebar_position: 2
---

# Chapter 11: Imitation Learning and Vision-Language-Action Models

## Learning Goals

- Understand vision-language-action models
- Learn imitation learning techniques
- Master multi-modal learning
- Implement VLA-based robot control
- Train imitation learning models
- Control robots using vision and language inputs

## Introduction to Imitation Learning

Imitation learning is a powerful approach where robots learn to perform tasks by observing demonstrations from human experts or other agents. Rather than learning from rewards like in reinforcement learning, imitation learning learns directly from expert demonstrations, making it particularly effective for complex tasks where defining appropriate reward functions is challenging.

### Why Imitation Learning?

Traditional reinforcement learning can struggle with sparse rewards and complex task specifications. Imitation learning addresses these challenges by:

1. **Direct Learning**: Learning directly from expert demonstrations
2. **Sample Efficiency**: Often requiring fewer samples than RL
3. **Natural Supervision**: Humans naturally provide demonstrations
4. **Complex Behaviors**: Learning complex, multi-step tasks effectively

### Key Approaches to Imitation Learning

1. **Behavior Cloning**: Direct supervised learning from demonstrations
2. **Inverse Reinforcement Learning**: Learning the reward function from demonstrations
3. **Generative Adversarial Imitation Learning (GAIL)**: Using adversarial training

## Behavior Cloning

Behavior cloning is the simplest form of imitation learning, treating the problem as a supervised learning task:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class RobotStateActionDataset(Dataset):
    def __init__(self, states, actions):
        """
        Dataset for robot state-action pairs
        states: List of state vectors
        actions: List of corresponding actions
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class BehaviorCloningNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(BehaviorCloningNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class BehaviorCloningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = BehaviorCloningNet(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, dataset, epochs=100, batch_size=32):
        """Train the behavior cloning network"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.network.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for states, actions in dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)

                predicted_actions = self.network(states)
                loss = self.criterion(predicted_actions, actions)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

        return losses

    def predict(self, state):
        """Predict action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.network(state_tensor).cpu().numpy()[0]
        return action


# Example: Train behavior cloning on simple navigation data
def generate_demo_data(num_demos=1000):
    """Generate demonstration data for a simple navigation task"""
    states = []
    actions = []

    for _ in range(num_demos):
        # Simple navigation: robot at [x, y], target at [tx, ty]
        robot_pos = np.random.uniform(-5, 5, 2)
        target_pos = np.random.uniform(-4, 4, 2)

        # State: relative position to target
        state = np.concatenate([robot_pos, target_pos, target_pos - robot_pos])  # 6-dimensional state

        # Action: move towards target (normalized)
        direction = target_pos - robot_pos
        action = direction / (np.linalg.norm(direction) + 1e-8)  # Normalize

        states.append(state)
        actions.append(action)

    return np.array(states), np.array(actions)


def main_behavior_cloning():
    # Generate demonstration data
    states, actions = generate_demo_data(num_demos=2000)

    # Create dataset
    dataset = RobotStateActionDataset(states, actions)

    # Initialize and train agent
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    agent = BehaviorCloningAgent(state_dim, action_dim)

    print("Training behavior cloning agent...")
    losses = agent.train(dataset, epochs=100, batch_size=64)

    # Test the trained agent
    test_robot_pos = np.array([1.0, 1.0])
    test_target_pos = np.array([-2.0, -2.0])
    test_state = np.concatenate([test_robot_pos, test_target_pos, test_target_pos - test_robot_pos])

    predicted_action = agent.predict(test_state)
    optimal_action = (test_target_pos - test_robot_pos) / np.linalg.norm(test_target_pos - test_robot_pos)

    print(f"\nTest case:")
    print(f"Robot position: {test_robot_pos}")
    print(f"Target position: {test_target_pos}")
    print(f"Optimal action: {optimal_action}")
    print(f"Predicted action: {predicted_action}")
    print(f"Action similarity: {np.dot(predicted_action, optimal_action):.3f}")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Behavior Cloning Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main_behavior_cloning()
```

## Generative Adversarial Imitation Learning (GAIL)

GAIL uses adversarial training to learn policies that are indistinguishable from expert demonstrations:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import gym


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return torch.tanh(self.network(state))  # Actions in [-1, 1]


class GAILAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Networks
        self.discriminator = Discriminator(state_dim, action_dim).to(self.device)
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)

        # Optimizers
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.bce_loss = nn.BCELoss()

    def update_discriminator(self, expert_states, expert_actions, policy_states, policy_actions):
        """Update discriminator to distinguish expert vs policy trajectories"""
        # Labels: 1 for expert, 0 for policy
        expert_labels = torch.ones(expert_states.size(0), 1).to(self.device)
        policy_labels = torch.zeros(policy_states.size(0), 1).to(self.device)

        # Discriminator loss
        expert_logits = self.discriminator(expert_states, expert_actions)
        policy_logits = self.discriminator(policy_states, policy_actions.detach())

        expert_loss = self.bce_loss(expert_logits, expert_labels)
        policy_loss = self.bce_loss(policy_logits, policy_labels)

        disc_loss = expert_loss + policy_loss

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        return disc_loss.item()

    def update_policy(self, states):
        """Update policy to fool discriminator"""
        actions = self.policy(states)
        logits = self.discriminator(states, actions)

        # Policy loss: maximize log(D(s,a)) to fool discriminator
        policy_loss = -torch.log(logits + 1e-8).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()

    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy(state_tensor).cpu().numpy()[0]
        return action


def train_gail_example():
    """Example of training GAIL on a simple environment"""
    import gym

    # Create environment
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Generate expert demonstrations (using random policy for this example)
    # In practice, these would come from expert demonstrations
    expert_states = []
    expert_actions = []

    # For demonstration, we'll create "expert" data using a slightly better random policy
    for episode in range(100):
        state = env.reset()
        if isinstance(state, tuple):  # Handle newer gym versions
            state = state[0]

        for step in range(100):
            # "Expert" action: slightly biased towards center
            action = env.action_space.sample()
            if state[0] > 0:  # If angle is positive, apply negative torque
                action[0] = max(action[0], -1.0)
            else:  # If angle is negative, apply positive torque
                action[0] = min(action[0], 1.0)

            expert_states.append(state.copy())
            expert_actions.append(action.copy())

            result = env.step(action)
            if len(result) == 4:
                state, reward, done, info = result
            else:
                state, reward, terminated, truncated, info = result
                done = terminated or truncated

            if done:
                break

    # Convert to tensors
    expert_states = torch.FloatTensor(expert_states)
    expert_actions = torch.FloatTensor(expert_actions)

    # Initialize GAIL agent
    agent = GAILAgent(state_dim, action_dim)

    # Training loop
    episodes = 1000
    for episode in range(episodes):
        # Collect policy rollout
        policy_states = []
        policy_actions = []

        state = env.reset()
        if isinstance(state, tuple):  # Handle newer gym versions
            state = state[0]

        for step in range(50):  # Short rollout for efficiency
            action = agent.select_action(state)
            policy_states.append(state.copy())
            policy_actions.append(action.copy())

            result = env.step(action)
            if len(result) == 4:
                state, reward, done, info = result
            else:
                state, reward, terminated, truncated, info = result
                done = terminated or truncated

            if done:
                break

        if len(policy_states) == 0:
            continue

        # Convert to tensors
        policy_states = torch.FloatTensor(policy_states)
        policy_actions = torch.FloatTensor(policy_actions)

        # Update discriminator and policy
        disc_loss = agent.update_discriminator(
            expert_states, expert_actions,
            policy_states, policy_actions
        )

        policy_loss = agent.update_policy(policy_states)

        if episode % 100 == 0:
            print(f'Episode {episode}, Disc Loss: {disc_loss:.4f}, Policy Loss: {policy_loss:.4f}')


if __name__ == '__main__':
    print("Training GAIL agent...")
    train_gail_example()
```

## Vision-Language-Action (VLA) Models

VLA models integrate vision, language, and action understanding into unified architectures:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import clip
from PIL import Image
import torchvision.transforms as transforms


class VLATransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(VLATransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x


class VisionEncoder(nn.Module):
    def __init__(self, input_channels=3, embed_dim=512):
        super(VisionEncoder, self).__init__()

        # Simple CNN for vision encoding
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        # Calculate flattened size
        conv_out_size = 64 * 7 * 7  # After convolutions on 64x64 image

        self.projection = nn.Linear(conv_out_size, embed_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.projection(x)
        return x


class LanguageEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, max_length=50):
        super(LanguageEncoder, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.embed_dim = embed_dim

        # Transformer blocks for language processing
        self.transformer_blocks = nn.ModuleList([
            VLATransformerBlock(embed_dim, 8, embed_dim * 4)
            for _ in range(4)
        ])

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len).expand(x.size(0), -1).to(x.device)

        x = self.token_embedding(x) + self.position_embedding(positions)

        for block in self.transformer_blocks:
            x = block(x)

        return x


class ActionDecoder(nn.Module):
    def __init__(self, embed_dim=512, action_dim=4):
        super(ActionDecoder, self).__init__()

        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, action_dim)
        )

    def forward(self, x):
        # Take the mean across sequence dimension
        x = x.mean(dim=1)
        action = self.action_head(x)
        return action


class VLAModel(nn.Module):
    def __init__(self, vocab_size=10000, action_dim=4, embed_dim=512):
        super(VLAModel, self).__init__()

        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.language_encoder = LanguageEncoder(vocab_size, embed_dim=embed_dim)
        self.action_decoder = ActionDecoder(embed_dim=embed_dim, action_dim=action_dim)

        # Cross-modal attention blocks
        self.cross_attention_blocks = nn.ModuleList([
            VLATransformerBlock(embed_dim, 8, embed_dim * 4)
            for _ in range(2)
        ])

        self.embed_dim = embed_dim

    def forward(self, image, text_tokens):
        # Encode vision and language
        vision_embeds = self.vision_encoder(image).unsqueeze(1)  # Add sequence dimension
        lang_embeds = self.language_encoder(text_tokens)

        # Concatenate vision and language embeddings
        combined_embeds = torch.cat([vision_embeds, lang_embeds], dim=1)

        # Cross-modal processing
        for block in self.cross_attention_blocks:
            combined_embeds = block(combined_embeds)

        # Decode to action
        action = self.action_decoder(combined_embeds)

        return action


class VLAProcessor:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

        # Initialize image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def tokenize_text(self, text, max_length=50):
        """Simple tokenization for demonstration"""
        # In practice, you'd use a proper tokenizer
        # For now, we'll use a simple approach
        words = text.lower().split()
        # Simple vocabulary mapping (in practice, use proper tokenizer)
        vocab = {"<pad>": 0, "<unk>": 1, "go": 2, "stop": 3, "forward": 4, "backward": 5,
                 "left": 6, "right": 7, "pick": 8, "place": 9, "object": 10, "red": 11,
                 "blue": 12, "green": 13, "box": 14, "ball": 15, "table": 16}

        tokens = []
        for word in words[:max_length]:
            tokens.append(vocab.get(word, vocab["<unk>"]))

        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(vocab["<pad>"])

        return np.array(tokens)

    def process(self, image, instruction):
        """Process image and instruction to generate action"""
        # Preprocess image
        if isinstance(image, str):  # If image path provided
            image = Image.open(image)
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        # Tokenize instruction
        text_tokens = self.tokenize_text(instruction)
        text_tensor = torch.LongTensor(text_tokens).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            action = self.model(image_tensor, text_tensor)

        return action.cpu().numpy()[0]


def train_vla_example():
    """Example of training a VLA model"""
    # Initialize model
    vocab_size = 10000
    action_dim = 4  # [vx, vy, vz, rotation]
    model = VLAModel(vocab_size=vocab_size, action_dim=action_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Create dummy training data
    batch_size = 8
    image_height, image_width = 64, 64
    max_text_length = 20
    action_dim = 4

    # Dummy data for training
    images = torch.randn(batch_size, 3, image_height, image_width).to(device)
    text_tokens = torch.randint(0, vocab_size, (batch_size, max_text_length)).to(device)
    target_actions = torch.randn(batch_size, action_dim).to(device)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        predicted_actions = model(images, text_tokens)

        # Compute loss
        loss = criterion(predicted_actions, target_actions)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    print(f'Final loss: {loss.item():.4f}')

    # Test the trained model
    processor = VLAProcessor(model, device)

    # Create dummy image (random)
    dummy_image = torch.rand(3, 64, 64)
    instruction = "go forward and pick the red ball"

    try:
        action = processor.process(dummy_image, instruction)
        print(f"Generated action for '{instruction}': {action}")
    except Exception as e:
        print(f"Error during processing: {e}")

    return model


if __name__ == '__main__':
    print("Training VLA model...")
    trained_model = train_vla_example()
```

## Multi-Modal Learning Integration

### Combining Vision, Language, and Action

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MultiModalFusion(nn.Module):
    def __init__(self, vision_dim, language_dim, action_dim, fusion_dim=256):
        super(MultiModalFusion, self).__init__()

        # Individual modality encoders
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )

        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )

        # Fusion mechanism
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU()
        )

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, action_dim)
        )

    def forward(self, vision_features, language_features):
        # Encode individual modalities
        vision_encoded = self.vision_encoder(vision_features)
        language_encoded = self.language_encoder(language_features)

        # Concatenate and fuse
        concatenated = torch.cat([vision_encoded, language_encoded], dim=1)
        fused = self.fusion(concatenated)

        # Decode to action
        action = self.action_decoder(fused)

        return action


class MultiModalRobotController:
    def __init__(self, vision_dim=512, language_dim=512, action_dim=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.controller = MultiModalFusion(
            vision_dim=vision_dim,
            language_dim=language_dim,
            action_dim=action_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.controller.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train_batch(self, vision_batch, language_batch, action_batch):
        """Train on a batch of multi-modal data"""
        vision_batch = torch.FloatTensor(vision_batch).to(self.device)
        language_batch = torch.FloatTensor(language_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)

        # Forward pass
        predicted_actions = self.controller(vision_batch, language_batch)

        # Compute loss
        loss = self.criterion(predicted_actions, action_batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict_action(self, vision_features, language_features):
        """Predict action given vision and language inputs"""
        vision_tensor = torch.FloatTensor(vision_features).unsqueeze(0).to(self.device)
        language_tensor = torch.FloatTensor(language_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.controller(vision_tensor, language_tensor)

        return action.cpu().numpy()[0]


def generate_multimodal_demo_data(num_samples=1000):
    """Generate demonstration data for multi-modal learning"""
    vision_features = []
    language_features = []
    actions = []

    for _ in range(num_samples):
        # Vision features: object positions, colors, etc.
        vision_feat = np.random.randn(512)

        # Language features: encoded instruction
        lang_feat = np.random.randn(512)

        # Action: movement command
        action = np.random.randn(4)

        # Make action somewhat related to features (for learning signal)
        # This simulates a real relationship in demonstration data
        action[0] = vision_feat[0] * 0.1 + lang_feat[10] * 0.1  # Example relationship
        action[1] = vision_feat[50] * 0.1 + lang_feat[20] * 0.1

        vision_features.append(vision_feat)
        language_features.append(lang_feat)
        actions.append(action)

    return np.array(vision_features), np.array(language_features), np.array(actions)


def train_multimodal_controller():
    """Train multi-modal controller"""
    # Generate demonstration data
    vision_data, language_data, action_data = generate_multimodal_demo_data(num_samples=2000)

    # Initialize controller
    controller = MultiModalRobotController(vision_dim=512, language_dim=512, action_dim=4)

    # Training
    batch_size = 32
    epochs = 100

    for epoch in range(epochs):
        total_loss = 0

        # Shuffle data
        indices = np.random.permutation(len(vision_data))

        for i in range(0, len(vision_data), batch_size):
            batch_indices = indices[i:i+batch_size]

            vision_batch = vision_data[batch_indices]
            language_batch = language_data[batch_indices]
            action_batch = action_data[batch_indices]

            loss = controller.train_batch(vision_batch, language_batch, action_batch)
            total_loss += loss

        avg_loss = total_loss / (len(vision_data) // batch_size)

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')

    # Test the controller
    test_vision = np.random.randn(512)
    test_language = np.random.randn(512)
    predicted_action = controller.predict_action(test_vision, test_language)

    print(f"Test prediction - Vision shape: {test_vision.shape}, Language shape: {test_language.shape}")
    print(f"Predicted action: {predicted_action}")

    return controller


if __name__ == '__main__':
    print("Training multi-modal controller...")
    controller = train_multimodal_controller()
```

## NVIDIA VLA Implementation

### Understanding NVIDIA's VLA Framework

NVIDIA's Vision-Language-Action (VLA) models represent state-of-the-art approaches to multimodal robotics:

```python
import torch
import torch.nn as nn
import numpy as np


class NVidiaVLAPolicy(nn.Module):
    def __init__(self, vision_feature_dim=512, language_feature_dim=512, proprioception_dim=10, action_dim=7):
        """
        NVIDIA-style VLA policy that combines vision, language, and proprioception
        vision_feature_dim: Dimension of visual features
        language_feature_dim: Dimension of language features
        proprioception_dim: Dimension of robot proprioception (joint angles, etc.)
        action_dim: Dimension of action space (e.g., 7-DoF for manipulation)
        """
        super(NVidiaVLAPolicy, self).__init__()

        # Feature encoders
        self.vision_encoder = self._create_feature_encoder(vision_feature_dim, 256)
        self.language_encoder = self._create_feature_encoder(language_feature_dim, 256)
        self.proprio_encoder = self._create_feature_encoder(proprioception_dim, 64)

        # Late fusion transformer
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,  # Combined feature dimension
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Task conditioning
        self.task_embedding = nn.Parameter(torch.randn(1, 1, 512))

    def _create_feature_encoder(self, input_dim, output_dim):
        """Create a feature encoder network"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, vision_features, language_features, proprioception):
        """
        Forward pass of VLA policy
        vision_features: [batch_size, vision_feature_dim]
        language_features: [batch_size, language_feature_dim]
        proprioception: [batch_size, proprioception_dim]
        """
        batch_size = vision_features.size(0)

        # Encode features
        vision_enc = self.vision_encoder(vision_features).unsqueeze(1)  # [B, 1, D]
        lang_enc = self.language_encoder(language_features).unsqueeze(1)  # [B, 1, D]
        proprio_enc = self.proprio_encoder(proprioception).unsqueeze(1)  # [B, 1, D]

        # Concatenate all modalities plus task conditioning
        fused_features = torch.cat([
            self.task_embedding.expand(batch_size, -1, -1),  # Task conditioning
            vision_enc,
            lang_enc,
            proprio_enc
        ], dim=1)  # [B, 4, 512]

        # Apply transformer
        attended_features = self.fusion_transformer(fused_features)

        # Use the first token (task-conditioned representation) for action prediction
        task_repr = attended_features[:, 0, :]  # [B, 512]

        # Decode to action
        action = self.action_decoder(task_repr)

        return action


class NVidiaVLAProcessor:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

    def encode_language(self, instruction):
        """Encode natural language instruction into features"""
        # In practice, this would use a pre-trained language model like BERT or CLIP
        # For demonstration, we'll use a simple approach
        vocab = {"<pad>": 0, "<unk>": 1, "pick": 2, "place": 3, "go": 4, "to": 5,
                 "the": 6, "red": 7, "blue": 8, "box": 9, "ball": 10, "table": 11}

        tokens = instruction.lower().split()
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

        # Simple embedding based on token IDs (in practice, use pre-trained embeddings)
        embeddings = np.zeros(512)
        for i, token_id in enumerate(token_ids):
            embeddings[i % 512] += token_id * 0.1

        return embeddings

    def encode_vision(self, image_features):
        """Encode visual features (in practice, from a vision model)"""
        # In practice, this would come from a pre-trained vision model
        # For demonstration, we'll just normalize the input
        return image_features / (np.linalg.norm(image_features) + 1e-8)

    def encode_proprioception(self, joint_angles, ee_pose):
        """Encode proprioceptive information"""
        # Combine joint angles and end-effector pose
        proprioception = np.concatenate([joint_angles, ee_pose])
        return proprioception / (np.linalg.norm(proprioception) + 1e-8)

    def predict_action(self, image_features, instruction, joint_angles, ee_pose):
        """Predict action given all modalities"""
        # Encode all inputs
        vision_encoded = self.encode_vision(image_features)
        language_encoded = self.encode_language(instruction)
        proprio_encoded = self.encode_proprioception(joint_angles, ee_pose)

        # Convert to tensors
        vision_tensor = torch.FloatTensor(vision_encoded).unsqueeze(0).to(self.device)
        language_tensor = torch.FloatTensor(language_encoded).unsqueeze(0).to(self.device)
        proprio_tensor = torch.FloatTensor(proprio_encoded).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            action = self.model(vision_tensor, language_tensor, proprio_tensor)

        return action.cpu().numpy()[0]


def create_nvidia_vla_demo():
    """Create and demonstrate NVIDIA-style VLA model"""
    # Initialize model
    model = NVidiaVLAPolicy(
        vision_feature_dim=512,
        language_feature_dim=512,
        proprioception_dim=17,  # 7 joint angles + 6-dof pose + 4 extra
        action_dim=7  # 7-DoF action for manipulation
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize processor
    processor = NVidiaVLAProcessor(model, device)

    # Create dummy training data
    batch_size = 16
    vision_batch = torch.randn(batch_size, 512).to(device)
    language_batch = torch.randn(batch_size, 512).to(device)
    proprio_batch = torch.randn(batch_size, 17).to(device)
    action_batch = torch.randn(batch_size, 7).to(device)

    # Train the model (dummy training for demonstration)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("Training NVIDIA-style VLA model...")
    for epoch in range(50):  # Just a few epochs for demo
        optimizer.zero_grad()

        pred_actions = model(vision_batch, language_batch, proprio_batch)
        loss = criterion(pred_actions, action_batch)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    print("Training completed!")

    # Demonstrate the model
    print("\nDemonstrating VLA model:")

    # Dummy inputs
    dummy_image_features = np.random.randn(512)
    dummy_instruction = "pick the red box and place it on the table"
    dummy_joints = np.random.randn(7)
    dummy_ee_pose = np.random.randn(10)  # 6-dof pose + extra

    action = processor.predict_action(dummy_image_features, dummy_instruction, dummy_joints, dummy_ee_pose)
    print(f"Instruction: '{dummy_instruction}'")
    print(f"Predicted action: {action}")

    return model, processor


if __name__ == '__main__':
    model, processor = create_nvidia_vla_demo()
```

## Real-World VLA Implementation for Robotics

### Integration with ROS 2

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import numpy as np


class VLARobotController(Node):
    def __init__(self):
        super().__init__('vla_robot_controller')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.action_pub = self.create_publisher(Pose, '/target_pose', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/robot_command',
            self.command_callback,
            10
        )

        # Initialize components
        self.bridge = CvBridge()
        self.current_image = None
        self.current_joints = None
        self.current_command = None
        self.vla_model = None  # Will be initialized later

        # Initialize VLA model
        self.initialize_vla_model()

        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('VLA Robot Controller initialized')

    def initialize_vla_model(self):
        """Initialize the VLA model"""
        # In practice, load a pre-trained model
        # For demonstration, we'll create a simple model
        self.vla_model = NVidiaVLAPolicy(
            vision_feature_dim=512,
            language_feature_dim=512,
            proprioception_dim=17,
            action_dim=7
        )

        # Load pre-trained weights if available
        # self.vla_model.load_state_dict(torch.load('pretrained_vla_model.pth'))

        self.vla_processor = NVidiaVLAProcessor(self.vla_model)
        self.get_logger().info('VLA model initialized')

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process image to extract features (simplified)
            # In practice, this would run through a pre-trained vision model
            image_features = self.process_image_for_features(cv_image)
            self.current_image = image_features

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def joint_callback(self, msg):
        """Process joint states"""
        try:
            # Extract joint positions
            self.current_joints = np.array(msg.position)
        except Exception as e:
            self.get_logger().error(f'Error processing joints: {e}')

    def command_callback(self, msg):
        """Process natural language command"""
        self.current_command = msg.data
        self.get_logger().info(f'Received command: {msg.data}')

    def process_image_for_features(self, image):
        """Extract visual features from image (simplified)"""
        # In practice, this would run through a pre-trained vision model
        # For demonstration, we'll use simple features

        # Resize image and extract simple features
        import cv2
        resized = cv2.resize(image, (64, 64))

        # Simple feature extraction: color histograms, edges, etc.
        features = []

        # Color histogram features
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([resized], [i], None, [8], [0, 256])
            features.extend(hist.flatten())

        # Flatten image as features
        flat_image = resized.flatten()
        features.extend(flat_image[:400])  # Limit to keep dimension reasonable

        # Pad or truncate to fixed dimension
        features = np.array(features[:512])
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)), 'constant')

        return features

    def control_loop(self):
        """Main control loop"""
        if (self.current_image is not None and
            self.current_joints is not None and
            self.current_command is not None):

            try:
                # Prepare inputs for VLA model
                image_features = self.current_image
                instruction = self.current_command

                # Get end-effector pose (simplified)
                ee_pose = np.zeros(10)  # Placeholder

                # Predict action using VLA model
                action = self.vla_processor.predict_action(
                    image_features,
                    instruction,
                    self.current_joints,
                    ee_pose
                )

                # Convert action to robot command
                self.execute_action(action)

                # Clear command after execution
                self.current_command = None

            except Exception as e:
                self.get_logger().error(f'Error in control loop: {e}')

    def execute_action(self, action):
        """Execute the predicted action"""
        # Interpret action based on its meaning
        # This is a simplified example

        if len(action) >= 3:
            # Use first 3 dimensions as velocity command
            cmd_vel = Twist()
            cmd_vel.linear.x = float(action[0])
            cmd_vel.linear.y = float(action[1])
            cmd_vel.angular.z = float(action[2])

            self.cmd_pub.publish(cmd_vel)
            self.get_logger().info(f'Published velocity command: {cmd_vel}')

        if len(action) >= 7:
            # Use as target pose (simplified)
            target_pose = Pose()
            target_pose.position.x = float(action[3])
            target_pose.position.y = float(action[4])
            target_pose.position.z = float(action[5])

            # Simple orientation (in practice, would be more complex)
            target_pose.orientation.w = 1.0  # Default orientation

            self.action_pub.publish(target_pose)
            self.get_logger().info(f'Published target pose: {target_pose}')


def main(args=None):
    rclpy.init(args=args)
    vla_controller = VLARobotController()

    try:
        rclpy.spin(vla_controller)
    except KeyboardInterrupt:
        pass
    finally:
        vla_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Imitation Learning for VLA Systems

### Combining Imitation Learning with VLA

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class VLAImitationDataset:
    def __init__(self, demonstrations):
        """
        Dataset for VLA imitation learning
        demonstrations: List of dicts with keys: 'vision', 'language', 'action', 'proprio'
        """
        self.demonstrations = demonstrations

    def __len__(self):
        return len(self.demonstrations)

    def __getitem__(self, idx):
        demo = self.demonstrations[idx]
        return {
            'vision': torch.FloatTensor(demo['vision']),
            'language': torch.FloatTensor(demo['language']),
            'proprio': torch.FloatTensor(demo['proprio']),
            'action': torch.FloatTensor(demo['action'])
        }


class VLAImitationLearner:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            vision = batch['vision'].to(self.device)
            language = batch['language'].to(self.device)
            proprio = batch['proprio'].to(self.device)
            actions = batch['action'].to(self.device)

            # Forward pass
            predicted_actions = self.model(vision, language, proprio)
            loss = self.criterion(predicted_actions, actions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                vision = batch['vision'].to(self.device)
                language = batch['language'].to(self.device)
                proprio = batch['proprio'].to(self.device)
                actions = batch['action'].to(self.device)

                predicted_actions = self.model(vision, language, proprio)
                loss = self.criterion(predicted_actions, actions)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches


def generate_vla_demonstrations(num_demos=1000):
    """Generate synthetic VLA demonstrations"""
    demonstrations = []

    for _ in range(num_demos):
        # Vision features (simulated)
        vision_features = np.random.randn(512)

        # Language features (simulated instruction encoding)
        language_features = np.random.randn(512)

        # Proprioception (joint angles, etc.)
        proprio_features = np.random.randn(17)

        # Action (expert demonstration)
        action = np.random.randn(7)

        # Create a simple relationship for learning signal
        action[0] = (vision_features[0] + language_features[10] + proprio_features[0]) * 0.1
        action[1] = (vision_features[100] + language_features[20] + proprio_features[5]) * 0.1

        demo = {
            'vision': vision_features,
            'language': language_features,
            'proprio': proprio_features,
            'action': action
        }

        demonstrations.append(demo)

    return demonstrations


def train_vla_imitation():
    """Train VLA model using imitation learning"""
    # Generate demonstrations
    print("Generating demonstrations...")
    demonstrations = generate_vla_demonstrations(num_demos=2000)

    # Create dataset and dataloader
    dataset = VLAImitationDataset(demonstrations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    print("Initializing VLA model...")
    model = NVidiaVLAPolicy(
        vision_feature_dim=512,
        language_feature_dim=512,
        proprioception_dim=17,
        action_dim=7
    )

    # Initialize learner
    learner = VLAImitationLearner(model)

    # Training loop
    print("Starting training...")
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = learner.train_epoch(dataloader)

        if epoch % 20 == 0:
            eval_loss = learner.evaluate(dataloader)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}')

    print("Training completed!")

    # Test the trained model
    test_demo = demonstrations[0]
    model.eval()
    with torch.no_grad():
        vision_tensor = torch.FloatTensor(test_demo['vision']).unsqueeze(0).to(learner.device)
        language_tensor = torch.FloatTensor(test_demo['language']).unsqueeze(0).to(learner.device)
        proprio_tensor = torch.FloatTensor(test_demo['proprio']).unsqueeze(0).to(learner.device)

        predicted_action = model(vision_tensor, language_tensor, proprio_tensor)
        actual_action = test_demo['action']

        print(f"Test - Actual action: {actual_action}")
        print(f"Test - Predicted action: {predicted_action.cpu().numpy()[0]}")
        print(f"Similarity: {np.dot(actual_action, predicted_action.cpu().numpy()[0]):.3f}")

    return model


if __name__ == '__main__':
    trained_model = train_vla_imitation()
```

## Hands-On Lab: VLA-Based Robot Control System

### Objective
Create a complete VLA-based robot control system that accepts natural language commands and executes them using visual and proprioceptive feedback.

### Prerequisites
- Completed Chapter 1-11
- ROS 2 Humble with Gazebo
- PyTorch installed
- Basic understanding of VLA concepts

### Steps

1. **Create a VLA lab package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python vla_robot_lab --dependencies rclpy sensor_msgs geometry_msgs std_msgs cv_bridge torch numpy matplotlib
   ```

2. **Create the main VLA control node** (`vla_robot_lab/vla_robot_lab/vla_control_node.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, JointState
   from geometry_msgs.msg import Twist, Pose
   from std_msgs.msg import String, Bool
   from cv_bridge import CvBridge
   import torch
   import numpy as np
   import json
   import time


   class VLABehaviorCloning(nn.Module):
       def __init__(self, vision_dim=512, language_dim=512, proprio_dim=17, action_dim=7):
           super(VLABehaviorCloning, self).__init__()

           # Modality encoders
           self.vision_encoder = nn.Sequential(
               nn.Linear(vision_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 256)
           )

           self.language_encoder = nn.Sequential(
               nn.Linear(language_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 256)
           )

           self.proprio_encoder = nn.Sequential(
               nn.Linear(proprio_dim, 64),
               nn.ReLU(),
               nn.Linear(64, 64)
           )

           # Fusion network
           self.fusion = nn.Sequential(
               nn.Linear(256 + 256 + 64, 512),
               nn.ReLU(),
               nn.Dropout(0.1),
               nn.Linear(512, 256),
               nn.ReLU(),
               nn.Linear(256, 128),
               nn.ReLU()
           )

           # Action decoder
           self.action_decoder = nn.Linear(128, action_dim)

       def forward(self, vision, language, proprio):
           vision_enc = self.vision_encoder(vision)
           language_enc = self.language_encoder(language)
           proprio_enc = self.proprio_encoder(proprio)

           fused = torch.cat([vision_enc, language_enc, proprio_enc], dim=1)
           features = self.fusion(fused)
           action = self.action_decoder(features)

           return action


   class VLAControlNode(Node):
       def __init__(self):
           super().__init__('vla_control_node')

           # Publishers
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.target_pose_pub = self.create_publisher(Pose, '/target_pose', 10)
           self.status_pub = self.create_publisher(Bool, '/vla_active', 10)

           # Subscribers
           self.image_sub = self.create_subscription(
               Image,
               '/camera/image_raw',
               self.image_callback,
               10
           )

           self.joint_sub = self.create_subscription(
               JointState,
               '/joint_states',
               self.joint_callback,
               10
           )

           self.command_sub = self.create_subscription(
               String,
               '/natural_language_command',
               self.command_callback,
               10
           )

           # Initialize components
           self.bridge = CvBridge()
           self.current_image = None
           self.current_joints = None
           self.current_command = None
           self.vla_model = None
           self.vla_active = False

           # Initialize VLA model
           self.initialize_model()

           # Control timer
           self.control_timer = self.create_timer(0.1, self.control_loop)

           # Training mode
           self.training_mode = False
           self.demo_buffer = []

           self.get_logger().info('VLA Control Node initialized')

       def initialize_model(self):
           """Initialize the VLA model"""
           self.vla_model = VLABehaviorCloning(
               vision_dim=512,
               language_dim=512,
               proprio_dim=17,
               action_dim=7
           )

           # Initialize with random weights for demo
           # In practice, load pre-trained weights

           self.get_logger().info('VLA model initialized')
           self.vla_active = True

       def image_callback(self, msg):
           """Process incoming image"""
           try:
               cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

               # Extract visual features
               features = self.extract_visual_features(cv_image)
               self.current_image = features

           except Exception as e:
               self.get_logger().error(f'Error processing image: {e}')

       def joint_callback(self, msg):
           """Process joint states"""
           try:
               # Extract joint positions and velocities
               if len(msg.position) > 0:
                   # Combine position and velocity information
                   joint_data = list(msg.position)
                   if len(msg.velocity) == len(msg.position):
                       joint_data.extend(list(msg.velocity))
                   else:
                       joint_data.extend([0.0] * len(msg.position))  # Zero velocity if not available

                   # Pad or truncate to fixed size
                   while len(joint_data) < 17:
                       joint_data.append(0.0)
                   joint_data = joint_data[:17]

                   self.current_joints = np.array(joint_data)
           except Exception as e:
               self.get_logger().error(f'Error processing joints: {e}')

       def command_callback(self, msg):
           """Process natural language command"""
           self.current_command = msg.data
           self.get_logger().info(f'Received VLA command: {msg.data}')

       def extract_visual_features(self, image):
           """Extract visual features from image"""
           import cv2

           # Resize image
           resized = cv2.resize(image, (64, 64))

           # Extract simple features (in practice, use a pre-trained vision model)
           features = []

           # Color histogram
           for i in range(3):  # BGR
               hist = cv2.calcHist([resized], [i], None, [8], [0, 256])
               features.extend(hist.flatten())

           # Edge features
           gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
           edges = cv2.Canny(gray, 50, 150)
           edge_hist = cv2.calcHist([edges], [0], None, [4], [0, 256])
           features.extend(edge_hist.flatten())

           # Flatten image as features
           flat_img = resized.flatten()
           features.extend(flat_img[:300])  # Limit features

           # Pad or truncate to fixed dimension
           features = np.array(features[:512])
           if len(features) < 512:
               features = np.pad(features, (0, 512 - len(features)), 'constant')

           return features

       def encode_language(self, text):
           """Encode natural language to features"""
           # Simple encoding for demonstration
           # In practice, use a pre-trained language model
           vocab = {
               '<pad>': 0, '<unk>': 1, 'go': 2, 'to': 3, 'the': 4, 'and': 5,
               'pick': 6, 'place': 7, 'up': 8, 'down': 9, 'left': 10, 'right': 11,
               'forward': 12, 'backward': 13, 'stop': 14, 'red': 15, 'blue': 16,
               'green': 17, 'yellow': 18, 'box': 19, 'ball': 20, 'table': 21,
               'shelf': 22, 'cup': 23, 'object': 24
           }

           tokens = text.lower().split()
           encoded = np.zeros(512)

           for i, token in enumerate(tokens[:50]):  # Limit to first 50 tokens
               token_id = vocab.get(token, vocab['<unk>'])
               encoded[i % 512] += token_id * 0.1  # Simple embedding

           return encoded

       def control_loop(self):
           """Main control loop"""
           if not self.vla_active:
               return

           if (self.current_image is not None and
               self.current_joints is not None and
               self.current_command is not None):

               try:
                   # Encode inputs
                   vision_tensor = torch.FloatTensor(self.current_image).unsqueeze(0)
                   language_features = self.encode_language(self.current_command)
                   language_tensor = torch.FloatTensor(language_features).unsqueeze(0)
                   proprio_tensor = torch.FloatTensor(self.current_joints).unsqueeze(0)

                   # Get action from VLA model
                   with torch.no_grad():
                       action_tensor = self.vla_model(vision_tensor, language_tensor, proprio_tensor)
                       action = action_tensor.cpu().numpy()[0]

                   # Execute action
                   self.execute_vla_action(action)

                   # Log the interaction
                   self.get_logger().info(
                       f'VLA executed command: "{self.current_command}", '
                       f'action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]'
                   )

                   # Clear command after execution
                   self.current_command = None

                   # If in training mode, save demonstration
                   if self.training_mode:
                       self.save_demonstration(
                           self.current_image.copy(),
                           language_features.copy(),
                           self.current_joints.copy(),
                           action.copy()
                       )

               except Exception as e:
                   self.get_logger().error(f'Error in VLA control: {e}')

       def execute_vla_action(self, action):
           """Execute the VLA-predicted action"""
           # Interpret action vector
           # First 3 elements: velocity command
           cmd_vel = Twist()
           cmd_vel.linear.x = float(np.clip(action[0], -1.0, 1.0))
           cmd_vel.linear.y = float(np.clip(action[1], -1.0, 1.0))
           cmd_vel.angular.z = float(np.clip(action[2], -1.0, 1.0))

           # Next 4 elements: target pose (simplified)
           target_pose = Pose()
           target_pose.position.x = float(action[3]) if len(action) > 3 else 0.0
           target_pose.position.y = float(action[4]) if len(action) > 4 else 0.0
           target_pose.position.z = float(action[5]) if len(action) > 5 else 0.0
           target_pose.orientation.w = 1.0  # Default orientation

           # Publish commands
           self.cmd_vel_pub.publish(cmd_vel)
           self.target_pose_pub.publish(target_pose)

           # Publish status
           status_msg = Bool()
           status_msg.data = True
           self.status_pub.publish(status_msg)

       def save_demonstration(self, vision, language, proprio, action):
           """Save demonstration for imitation learning"""
           demo = {
               'vision': vision.tolist(),
               'language': language.tolist(),
               'proprio': proprio.tolist(),
               'action': action.tolist(),
               'timestamp': time.time()
           }

           self.demo_buffer.append(demo)

           # Limit buffer size
           if len(self.demo_buffer) > 1000:
               self.demo_buffer = self.demo_buffer[-500:]  # Keep last 500 demos

           self.get_logger().info(f'Demonstration saved. Buffer size: {len(self.demo_buffer)}')

       def start_training_mode(self):
           """Start collecting demonstrations"""
           self.training_mode = True
           self.get_logger().info('VLA training mode activated - collecting demonstrations')

       def stop_training_mode(self):
           """Stop collecting demonstrations"""
           self.training_mode = False
           self.get_logger().info('VLA training mode deactivated')

       def save_demonstrations_to_file(self, filename='vla_demonstrations.json'):
           """Save collected demonstrations to file"""
           with open(filename, 'w') as f:
               json.dump(self.demo_buffer, f)
           self.get_logger().info(f'Saved {len(self.demo_buffer)} demonstrations to {filename}')


   def main(args=None):
       rclpy.init(args=args)
       vla_control_node = VLAControlNode()

       try:
           # Start in training mode for demonstration
           def start_training_timer_callback():
               vla_control_node.start_training_mode()
               start_timer.cancel()

           start_timer = vla_control_node.create_timer(5.0, start_training_timer_callback)

           rclpy.spin(vla_control_node)
       except KeyboardInterrupt:
           pass
       finally:
           # Save demonstrations before shutting down
           vla_control_node.save_demonstrations_to_file()
           vla_control_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`vla_robot_lab/launch/vla_control.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='true',
           description='Use simulation (Gazebo) clock if true'
       )

       # VLA control node
       vla_control_node = Node(
           package='vla_robot_lab',
           executable='vla_control_node',
           name='vla_control_node',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
           output='screen'
       )

       return LaunchDescription([
           use_sim_time,
           vla_control_node
       ])
   ```

4. **Update setup.py**:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'vla_robot_lab'

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
       description='VLA robot lab for robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'vla_control_node = vla_robot_lab.vla_control_node:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select vla_robot_lab
   source install/setup.bash
   ```

6. **Run the VLA system**:
   ```bash
   ros2 launch vla_robot_lab vla_control.launch.py
   ```

### Expected Results
- The system should accept natural language commands via the `/natural_language_command` topic
- Visual and proprioceptive information should be processed to generate appropriate actions
- Demonstrations should be collected when in training mode
- The system should execute commands like "go to the red box" or "pick up the blue ball"

### Troubleshooting Tips
- Ensure PyTorch is properly installed
- Verify that camera and joint state topics are being published
- Check that the VLA model is properly initialized
- Monitor the logs for VLA processing status

## Summary

In this chapter, we've explored the cutting-edge field of Vision-Language-Action (VLA) models and imitation learning for robotics. We've covered:

1. **Imitation Learning Fundamentals**: Behavior cloning, GAIL, and their applications to robotics
2. **VLA Architecture**: How to combine vision, language, and action understanding in unified models
3. **Multi-Modal Integration**: Techniques for fusing different sensory modalities
4. **NVIDIA VLA Implementation**: Understanding state-of-the-art VLA frameworks
5. **Practical Integration**: How to implement VLA systems with ROS 2

The hands-on lab provided experience with creating a complete VLA-based robot control system that accepts natural language commands and executes them using visual and proprioceptive feedback. This represents one of the most advanced approaches to robot learning and control, bridging the gap between human intentions and robotic actions.

This foundation prepares us for the next chapter on Human-Robot Interaction, where we'll explore how to make these intelligent systems more intuitive and collaborative for human users.