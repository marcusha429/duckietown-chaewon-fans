import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sac.networks import Actor, Critic


class SACAgent:
   def __init__(self, obs_dim, act_dim, hidden_sizes):
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.act_dim = act_dim
       print(f"Using device: {self.device}")
       print(f"Action dimension: {act_dim}")
      
       # Actor network
       self.actor = Actor(obs_dim, act_dim, hidden_sizes).to(self.device)
       self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
      
       # Two Critic networks
       self.critic1 = Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
       self.critic2 = Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
       self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
       self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
      
       # Target networks
       self.target_critic1 = Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
       self.target_critic2 = Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
       self.target_critic1.load_state_dict(self.critic1.state_dict())
       self.target_critic2.load_state_dict(self.critic2.state_dict())
      
       # Temperature parameter
       self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
       self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
       self.target_entropy = -act_dim


       self.gamma = 0.99
       self.tau = 0.005


   @property
   def alpha(self):
       return self.log_alpha.exp()


   def _format_observation(self, obs):
       """Format observation to tensor with correct shape."""
       if isinstance(obs, np.ndarray):
           obs = torch.FloatTensor(obs)
       elif not isinstance(obs, torch.Tensor):
           obs = torch.FloatTensor([obs])
          
       # Ensure observation is 2D [batch_size, obs_dim]
       if obs.dim() == 1:
           obs = obs.unsqueeze(0)
          
       return obs.to(self.device)


   def select_action(self, obs):
       """
       Select action from the policy.
      
       Args:
           obs: numpy array or tensor of shape (obs_dim,)
          
       Returns:
           action: numpy array of shape (act_dim,)
       """
       with torch.no_grad():
           # Format observation
           obs_tensor = self._format_observation(obs)
          
           # Get action from policy
           action, _, _ = self.actor.sample(obs_tensor)
          
           # Convert to numpy and ensure shape
           if isinstance(action, torch.Tensor):
               action = action.cpu().numpy()
              
           # For single observations, return array of shape (act_dim,)
           if action.shape[0] == 1:
               action = action[0]
              
           # Ensure action has correct shape and range
           action = np.clip(action, -1.0, 1.0)
           assert action.shape == (self.act_dim,), f"Invalid action shape: {action.shape}"
          
       return action


   def update_parameters(self, batch, gradient_steps=1):
       """Update agent parameters using batch of experience."""
       for _ in range(gradient_steps):
           # Convert batch to tensors with [batch_size, dim] shape
           obs = self._format_observation(batch['obs'])
           next_obs = self._format_observation(batch['next_obs'])
           acts = torch.FloatTensor(batch['acts']).to(self.device)
           rews = torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device)
           done = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)


           # Sample actions and compute log probs
           new_actions, log_prob, _ = self.actor.sample(obs)
          
           # Compute critic loss
           current_q1 = self.critic1(obs, acts)
           current_q2 = self.critic2(obs, acts)
          
           # Compute next state value
           with torch.no_grad():
               next_actions, next_log_prob, _ = self.actor.sample(next_obs)
               next_q1 = self.target_critic1(next_obs, next_actions)
               next_q2 = self.target_critic2(next_obs, next_actions)
               next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
               target_q = rews + self.gamma * (1 - done) * next_q


           # Compute critic losses
           critic1_loss = F.mse_loss(current_q1, target_q.detach())
           critic2_loss = F.mse_loss(current_q2, target_q.detach())


           # Update critics
           self.critic1_optimizer.zero_grad()
           critic1_loss.backward()
           torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
           self.critic1_optimizer.step()


           self.critic2_optimizer.zero_grad()
           critic2_loss.backward()
           torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
           self.critic2_optimizer.step()


           # Update policy
           q_new_actions = torch.min(
               self.critic1(obs, new_actions),
               self.critic2(obs, new_actions)
           )
           actor_loss = (self.alpha * log_prob - q_new_actions).mean()


           self.actor_optimizer.zero_grad()
           actor_loss.backward()
           torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
           self.actor_optimizer.step()


           # Update temperature parameter
           alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
           self.alpha_optimizer.zero_grad()
           alpha_loss.backward()
           self.alpha_optimizer.step()


           # Update target networks
           for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
               target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
           for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
               target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
