import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
import torch.optim as optim

from customer_simulation.customer_simulation import CustomerSimulator

class DynamicPricingEnv(gym.Env):
    def __init__(self, data, features, target):
        super(DynamicPricingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.features = features
        self.target = target
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # Initialize the CustomerSimulator
        self.customer_simulator = CustomerSimulator('data_cleaning/Cleaned_Data_Categories.csv')
        
        # Define action space: Continuous action space for price multiplier
        self.action_space = spaces.Box(
            low=0.5,   # 50% of base price
            high=2.0,  # 200% of base price
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space: Based on feature dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)
    
    def _get_observation(self):
        obs = self.data.loc[self.current_step, self.features].values.astype(np.float32)
        return obs
    
    def step(self, action):
        # Get current data point
        current_data = self.data.loc[self.current_step]
        item_id = current_data['Item']
        base_price = current_data['Price']
        cost = current_data['Cost']
        date = current_data['Date'] + ' ' + current_data['Time']
        initial_quantity = current_data['Total Quantity']

        # Apply price adjustment
        price_multiplier = action[0]
        new_price = base_price * price_multiplier

        # Ensure new price does not exceed 200% of base price
        new_price = min(new_price, base_price * self.customer_simulator.MAX_PRICE_RATIO)

        # Simulate customer purchases using CustomerSimulator
        simulation_result = self.customer_simulator.simulate_day(
            item_id=item_id,
            price=new_price,
            date=date,
            initial_quantity=initial_quantity
        )

        total_sold = simulation_result['total_sold']
        total_revenue = simulation_result['total_revenue']
        remaining_quantity = simulation_result['remaining_quantity']

        # Calculate profit
        profit = (new_price - cost) * total_sold

        # Set reward as profit
        reward = profit

        # Move to the next time step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Get next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape)

        info = {
            'new_price': new_price,
            'total_sold': total_sold,
            'remaining_quantity': remaining_quantity,
            'total_revenue': total_revenue
        }
        return obs, reward, done, info
