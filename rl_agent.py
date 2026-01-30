"""
RL Agent Module for Airport AGV Charging Decisions

DQN-based RL agent for charging decisions:
- Input: state_vector (created by sim_model_RL.py)
- Output: action (0: no charge, 1: Ch1, 2: Ch2)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


class DQN(nn.Module):
    """DQN network for charging decision"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ChargingAgent:
    """
    RL Agent for charging decisions ONLY
    
    Action Space:
        0: No charging (go to depot directly)
        1: Charge at Ch1
        2: Charge at Ch2
    
    State Space (created by sim_model_RL.py):
        s = [SoC, Q_Ch1, Q_Ch2, Dist_Ch1, Dist_Ch2, Work_Time] (Dim=6)
    """
    
    def __init__(self, state_dim=6, action_dim=3, lr=0.001, gamma=0.98):
        """
        Args:
            state_dim: dimension of state (default 6)
            action_dim: dimension of action (default 3: no_charge, ch1, ch2)
            lr: learning rate (Increased to 0.001 for faster convergence)
            gamma: discount factor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # DQN networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_q_network = DQN(state_dim, action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = []
        self.batch_size = 32
        self.max_buffer_size = 10000
        
        # Exploration
        self.epsilon = 1.0
        # Per-episode decay: ~0.995 gives ~300 episodes of exploration
        self.epsilon_decay = 0.995 
        self.epsilon_min = 0.01
        
    def select_action(self, state_vector: np.ndarray) -> int:
        """
        Select charging action using epsilon-greedy policy
        
        Args:
            state_vector: 6-dim State vector calculated by sim_model_RL
            
        Returns:
            action: 0 (no charge), 1 (Ch1), 2 (Ch2)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values, dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)
    
    def update_epsilon(self):
        """Decay epsilon at the end of episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self):
        """Train DQN with experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        state_tensor = torch.FloatTensor(np.array(states))
        next_state_tensor = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(state_tensor).gather(1, actions).squeeze(1)
        
        # Next Q values from target network
        next_q_values = self.target_q_network(next_state_tensor).max(1)[0].detach()
        
        # TD target
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, path: str):
        """Save Q-network state dictionary"""
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load Q-network state dictionary"""
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval() # Set to evaluation mode
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        print(f"Model loaded from {path}")


# ==============================================================================
# Global Agent Instance (Singleton)
# ==============================================================================

_charging_agent = None


def get_charging_agent() -> ChargingAgent:
    """Get or create global charging agent"""
    global _charging_agent
    if _charging_agent is None:
        _charging_agent = ChargingAgent()
    return _charging_agent


def reset_agent():
    """Reset global agent (for testing)"""
    global _charging_agent
    _charging_agent = None


# ==============================================================================
# Testing
# ==============================================================================

if __name__ == "__main__":
    print("Testing RL Charging Agent...")
    
    agent = get_charging_agent()
    
    # Test with dummy state vector
    state_vec = np.array([0.25, 0.5, 0.3, 0.4, 0.2, 0.1, 0.5], dtype=np.float32)
    
    print(f"State vector shape: {state_vec.shape}")
    print(f"State vector: {state_vec}")
    
    action = agent.select_action(state_vec)
    print(f"Charging action: {action} (0=no, 1=Ch1, 2=Ch2)")
    
    print("\nAgent module test passed!")
