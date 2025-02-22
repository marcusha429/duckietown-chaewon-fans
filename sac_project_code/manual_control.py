#!/usr/bin/env python3

import sys
import time
import numpy as np
import gym_duckietown
from gym_duckietown.simulator import Simulator

env = Simulator(
    seed=1,
    map_name="loop_empty",
    domain_rand=0,
    camera_width=640,
    camera_height=480,
    accept_start_angle_deg=4,
    full_transparency=True,
    distortion=False,
)

# Initialize
env.reset()
env.render()

print("Control the Duckiebot using:")
print("W/UP = forward")
print("S/DOWN = backward")
print("A/LEFT = turn left")
print("D/RIGHT = turn right")
print("SPACE = stop")
print("Q = quit")

def get_key():
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Main loop
try:
    while True:
        key = get_key()
        
        action = np.array([0.0, 0.0])
        
        if key in ['w', 'W', '\x1b[A']:  # Forward
            action = np.array([0.44, 0.44])
        elif key in ['s', 'S', '\x1b[B']:  # Backward
            action = np.array([-0.44, -0.44])
        elif key in ['a', 'A', '\x1b[D']:  # Left
            action = np.array([0.35, -0.35])
        elif key in ['d', 'D', '\x1b[C']:  # Right
            action = np.array([-0.35, 0.35])
        elif key == ' ':  # Stop
            action = np.array([0.0, 0.0])
        elif key in ['q', 'Q']:  # Quit
            break
            
        obs, reward, done, info = env.step(action)
        env.render()
        
        print(f"Action: {action}, Reward: {reward:.3f}")
        
        if done:
            print("Episode finished! Resetting...")
            env.reset()
            
        time.sleep(0.1)  # Small delay to control speed

finally:
    env.close()