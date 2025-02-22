import numpy as np
import torch
import gym
from env.duckietown_env import DuckietownEnvWrapper
from sac.agent import SACAgent
from sac.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


def speed_steering_to_wheels(speed, steering):

    #speed = 0 -> 1
    #steering (+) -> left turn , steering (-) -> right turn

   left = speed - steering
   right = speed + steering
   return left, right


def clean_action(raw_action):
   """
        raw_action[speed, steering] but have to clean to numpy array for computing thing
   """
   # Convert to numpy array
   action = np.asarray(raw_action, dtype=np.float64)
  
   # Ensure we have exactly 2 elements and flatten
   if len(action.shape) > 1:
       action = action.flatten()
   if action.size != 2:
       raise ValueError(f"Action must have exactly 2 elements, got shape {action.shape}")
  
   # Clip values to valid range (couldn't go faster than max speed or turn sharper than max steering)
   action = np.clip(action, -1.0, 1.0)
  
   return action


def main():
   # Hyperparameters
   max_episodes = 500
   max_steps = 1000
   start_steps = 1000
   batch_size = 256
   update_after = 1000
   update_every = 50
   replay_size = 1000000


   print("\n=== Initializing Training ===")
   env = DuckietownEnvWrapper()
  
   obs_dim = env.observation_space.shape[0]
   act_dim = env.action_space.shape[0]
   print(f"\nEnvironment configured:")
   print(f"- Observation dimension: {obs_dim}")
   print(f"- Action dimension: {act_dim}")


   # Set seeds
   print("\n=== Setting Seeds ===")
   seed = 42
   env.seed(seed)
   torch.manual_seed(seed)
   np.random.seed(seed)
  
   # Initialize agent and replay buffer
   print("\n=== Initializing Agent ===")
   agent = SACAgent(obs_dim, act_dim, hidden_sizes=[256, 256])
   replay_buffer = ReplayBuffer(obs_dim, act_dim, size=replay_size)


   total_steps = 0
   rewards_history = []
  
   print("\n=== Starting Training ===")
   print("-" * 50)
  
   try:
       for episode in range(max_episodes):
           episode_seed = seed + episode
           print(f"\nEpisode {episode + 1}/{max_episodes}")
           print(f"Using seed: {episode_seed}")
          
           # Reset environment
           try:
               state, _ = env.reset(seed=episode_seed)
           except Exception as e:
               print(f"Error resetting environment: {str(e)}")
               continue
          
           if state is None:
               print("Env returned None state, skipping episode.")
               continue
          
           episode_reward = 0
           episode_steps = 0
          
           for t in range(max_steps):
               total_steps += 1
               episode_steps += 1


               # Select action
               if total_steps < start_steps:
                   # random action
                   raw_action = env.action_space.sample()
               else:
                   # Use the agent's policy
                   state_tensor = torch.FloatTensor(state).unsqueeze(0)
                   raw_action = agent.select_action(state_tensor)
                   raw_action = raw_action  # shape (2,)


               # Clean to shape (2,) float64
               try:
                   action = clean_action(raw_action)
               except Exception as e:
                   print(f"Error cleaning action: {str(e)}")
                   break
              
               # Step the environment
               try:
                   next_state, reward, terminated, truncated, info = env.step(action)
               except Exception as e:
                   print(f"Error during step: {str(e)}")
                   print(f"Raw action was: {raw_action}")
                   break
              
               done = terminated or truncated
               episode_reward += reward


               # Store transition
               replay_buffer.store(state, action, reward, next_state, done)
               state = next_state


               # Update agent
               if total_steps >= update_after and total_steps % update_every == 0:
                   for _ in range(update_every):
                       batch = replay_buffer.sample_batch(batch_size)
                       agent.update_parameters(batch)


               if done:
                   break


           # Episode complete
           rewards_history.append(episode_reward)
           print(f"Episode {episode + 1}: Steps: {episode_steps}, Reward: {episode_reward:.2f}")
          
           if (episode + 1) % 10 == 0:
               avg_reward = np.mean(rewards_history[-10:])
               print(f"\nLast 10 episodes average reward: {avg_reward:.2f}")


   except KeyboardInterrupt:
       print("\nTraining interrupted by user.")
   except Exception as e:
       print(f"\nTraining stopped due to error: {str(e)}")
   finally:
       print("\nTraining finished")
       print(f"Total episodes completed: {len(rewards_history)}")
       if rewards_history:
           print(f"Final average reward: {np.mean(rewards_history[-10:]):.2f}")
          
           # Save training plot
           plt.figure(figsize=(10, 5))
           plt.plot(rewards_history)
           plt.xlabel('Episode')
           plt.ylabel('Reward')
           plt.title('Training Reward History')
           plt.grid(True)
           try:
               plt.savefig('training_rewards.png')
               print("Training plot saved as 'training_rewards.png'")
           except Exception as e:
               print(f"Could not save plot: {str(e)}")
           plt.close()


if __name__ == "__main__":
   main()