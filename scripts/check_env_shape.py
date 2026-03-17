from src.topdown.car_env import CarEnv
import numpy as np

try:
    env = CarEnv(render_mode=None)
    obs, info = env.reset()
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Actual observation shape: {obs.shape}")
    
    if env.observation_space.shape != obs.shape:
        print("MISMATCH DETECTED!")
    else:
        print("Shapes match.")
        
    # Check sensors count
    sim = env.simulation
    print(f"Sensor angles count: {len(sim.sensor_angles())}")
    
except Exception as e:
    print(f"Error during check: {e}")
