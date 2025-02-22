import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim
        
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
            
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super(Actor, self).__init__()
        self.act_dim = act_dim
        self.net = MLP(obs_dim, act_dim * 2, hidden_sizes)
        self.log_std_min = -20
        self.log_std_max = 2
    
    def _ensure_tensor_shape(self, x):
        """Ensure input tensor has batch dimension and correct type."""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor([x])
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.to(self.net.model[0].weight.device)
    
    def forward(self, obs):
        """
        Compute mean and standard deviation of the action distribution.
        Returns mean, std
        """
        # Ensure input tensor has correct shape
        obs = self._ensure_tensor_shape(obs)
        
        # Get mean and log_std
        net_out = self.net(obs)
        
        # Split output into mean and log_std
        mean, log_std = net_out.chunk(2, dim=-1)
        
        # Constrain log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # Scale to environment bounds
        velocity = (y_t[:, 0] + 1) * 0.25  # [-1, 1] -> [0, 0.5]
        steering = y_t[:, 1] * 0.2        # [-1, 1] -> [-0.2, 0.2]
        actions = torch.stack([velocity, steering], dim=-1)
        # Compute log_prob (unchanged)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return actions, log_prob, mu

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super(Critic, self).__init__()
        self.net = MLP(obs_dim + act_dim, 1, hidden_sizes)
    
    def _ensure_tensor_shape(self, x):
        """Ensure input tensor has batch dimension and correct type."""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor([x])
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.to(self.net.model[0].weight.device)
    
    def forward(self, obs, act):
        """
        Compute Q-value for state-action pair.
        Returns q_value
        """
        # Ensure inputs have correct shape
        obs = self._ensure_tensor_shape(obs)
        act = self._ensure_tensor_shape(act)
        
        # Concatenate along last dimension
        x = torch.cat([obs, act], dim=-1)
        
        return self.net(x)
