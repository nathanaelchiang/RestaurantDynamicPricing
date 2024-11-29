# ppo.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from customer_simulation import CustomerSimulator

class DynamicPricingEnv(gym.Env):
    def __init__(self, data):
        super(DynamicPricingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # Initialize the CustomerSimulator
        self.customer_simulator = CustomerSimulator('data_cleaning/Full_Dataset.csv')
        
        # Define action space: Percentage change in price, limited to ±10%
        self.action_space = spaces.Box(
            low=-0.1,   # -10%
            high=0.1,   # +10%
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space: Current price and remaining quantity
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )

        self.profit_history = []
        self.price_history = []
        self.item_ids = []
        self.time_stamps = []
        self.previous_price = None  # To keep track of the previous price
    
    def reset(self):
        self.current_step = 0
        self.previous_price = self.data.loc[self.current_step]['Price']
        obs = self._get_observation()
        return obs
    
    def _get_observation(self):
        current_data = self.data.loc[self.current_step]
        obs = np.array([
            self.previous_price,
            current_data['Quantity']
        ], dtype=np.float32)
        return obs
    
    def step(self, action):
        # Get current data point
        current_data = self.data.loc[self.current_step]
        item_id = current_data['Item']
        base_price = current_data['Price']
        cost = current_data['Cost']
        date = current_data['Date']
        time = current_data['Time']
        initial_quantity = current_data['Quantity']
    
        # Apply price adjustment
        price_change_percentage = action[0]  # This should be between -0.1 and 0.1
        new_price = self.previous_price * (1 + price_change_percentage)
    
        # Ensure new price does not exceed 200% of base price and not below 50% of base price
        new_price = min(new_price, base_price * self.customer_simulator.MAX_PRICE_RATIO)
        new_price = max(new_price, base_price * 0.5)
    
        # Combine date and time into a datetime object
        date_time_str = f"{date} {time}"
        date_time_obj = pd.to_datetime(date_time_str)
    
        # Simulate customer purchases using CustomerSimulator
        simulation_result = self.customer_simulator.simulate_day(
            item_id=item_id,
            price=new_price,
            date=date_time_obj,
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
    
        # Update previous price
        self.previous_price = new_price
    
        # Get next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
    
        info = {
            'new_price': new_price,
            'total_sold': total_sold,
            'remaining_quantity': remaining_quantity,
            'total_revenue': total_revenue,
            'profit': profit
        }

        self.profit_history.append(profit)
        self.price_history.append(new_price)
        self.item_ids.append(item_id)
        self.time_stamps.append(date_time_obj)

        return obs, reward, done, info

if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv('data_cleaning/Full_Dataset.csv')
    
    # For simplicity, we'll focus on a single item for training
    item_id = 106  # Replace with the item you want to focus on
    date_of_interest = '2022-01-01'  # The date we want to analyze
    
    # Filter data for the specific item and date
    item_data = data[(data['Item'] == item_id) & (data['Date'] == date_of_interest)]
    
    # Ensure that there is only one record per date/time for the item
    # Group by 'Date', 'Time', and 'Item' and aggregate
    item_data = item_data.groupby(['Date', 'Time', 'Item']).agg({
        'Price': 'mean',
        'Cost': 'mean',
        'Quantity': 'sum',
        # other fields as needed
    }).reset_index()
    
    # Sort data by time
    item_data['DateTime'] = pd.to_datetime(item_data['Date'] + ' ' + item_data['Time'])
    item_data = item_data.sort_values('DateTime').reset_index(drop=True)
    
    # Initialize the environment
    env = DynamicPricingEnv(item_data)
    env = DummyVecEnv([lambda: env])
    
    # Initialize the PPO agent
    model = PPO('MlpPolicy', env, verbose=1)
    
    # Train the agent
    model.learn(total_timesteps=5000)
    
    # Save the model
    model.save("ppo_dynamic_pricing")
    
    # Access the environment instance to get the stored data
    env_instance = env.envs[0]
    
    # Create a DataFrame from the stored data
    results_df = pd.DataFrame({
        'Time': env_instance.time_stamps,
        'Item_ID': env_instance.item_ids,
        'Price': env_instance.price_history,
        'Profit': env_instance.profit_history
    })
    
    # Sort by time
    results_df = results_df.sort_values('Time').reset_index(drop=True)
    
    # Save the results to a CSV file
    results_df.to_csv('price_time_data.csv', index=False)
    print("\nPrice and time data saved to 'price_time_data.csv'")
    
    # Plot price changes for the item over the course of the day
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Time'], results_df['Price'], marker='o', color='orange', label='Price')
    plt.title(f'Price Changes for Item ID {item_id} on {date_of_interest}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot total profit made during one day
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Time'], results_df['Profit'].cumsum(), marker='o', label='Cumulative Profit')
    plt.title(f'Cumulative Profit for Item ID {item_id} on {date_of_interest}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Profit')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
