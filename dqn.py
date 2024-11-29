# dqn.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from customer_simulation import CustomerSimulator


class DynamicPricingEnvDQN(gym.Env):
    def __init__(self, data, n_discrete_actions=21):
        super(DynamicPricingEnvDQN, self).__init__()

        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.max_steps = len(self.data) - 1

        # Initialize the CustomerSimulator
        self.customer_simulator = CustomerSimulator('data_cleaning/Full_Dataset.csv')

        # Define discrete action space with n_discrete_actions steps between -10% and +10%
        self.n_discrete_actions = n_discrete_actions
        self.action_space = spaces.Discrete(n_discrete_actions)

        # Define observation space: Current price and remaining quantity
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )

        # Calculate the step size for price adjustments
        self.price_adjustment_step = 0.2 / (n_discrete_actions - 1)  # Range from -10% to +10%

        # Initialize history tracking
        self.profit_history = []
        self.price_history = []
        self.item_ids = []
        self.time_stamps = []
        self.previous_price = None

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

    def _convert_discrete_to_continuous(self, discrete_action):
        """Convert discrete action index to continuous price adjustment percentage"""
        return -0.1 + (discrete_action * self.price_adjustment_step)

    def step(self, action):
        # Convert discrete action to continuous price adjustment
        price_change_percentage = self._convert_discrete_to_continuous(action)

        # Get current data point
        current_data = self.data.loc[self.current_step]
        item_id = current_data['Item']
        base_price = current_data['Price']
        cost = current_data['Cost']
        date = current_data['Date']
        time = current_data['Time']
        initial_quantity = current_data['Quantity']

        # Apply price adjustment
        new_price = self.previous_price * (1 + price_change_percentage)

        # Ensure new price doesn't exceed limits
        new_price = min(new_price, base_price * self.customer_simulator.MAX_PRICE_RATIO)
        new_price = max(new_price, base_price * 0.5)

        # Combine date and time
        date_time_str = f"{date} {time}"
        date_time_obj = pd.to_datetime(date_time_str)

        # Simulate customer behavior
        simulation_result = self.customer_simulator.simulate_day(
            item_id=item_id,
            price=new_price,
            date=date_time_obj,
            initial_quantity=initial_quantity
        )

        total_sold = simulation_result['total_sold']
        total_revenue = simulation_result['total_revenue']
        remaining_quantity = simulation_result['remaining_quantity']

        # Calculate profit and set as reward
        profit = (new_price - cost) * total_sold
        reward = profit

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Update previous price
        self.previous_price = new_price

        # Get next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Store history
        self.profit_history.append(profit)
        self.price_history.append(new_price)
        self.item_ids.append(item_id)
        self.time_stamps.append(date_time_obj)

        info = {
            'new_price': new_price,
            'total_sold': total_sold,
            'remaining_quantity': remaining_quantity,
            'total_revenue': total_revenue,
            'profit': profit
        }

        return obs, reward, done, info


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('data_cleaning/Full_Dataset.csv')

    # Setup parameters
    item_id = 106
    date_of_interest = '2022-01-01'

    # Filter and prepare data
    item_data = data[(data['Item'] == item_id) & (data['Date'] == date_of_interest)]
    item_data = item_data.groupby(['Date', 'Time', 'Item']).agg({
        'Price': 'mean',
        'Cost': 'mean',
        'Quantity': 'sum'
    }).reset_index()

    # Sort by time
    item_data['DateTime'] = pd.to_datetime(item_data['Date'] + ' ' + item_data['Time'])
    item_data = item_data.sort_values('DateTime').reset_index(drop=True)

    # Initialize environment
    env = DynamicPricingEnvDQN(item_data)
    env = DummyVecEnv([lambda: env])

    # Initialize DQN agent
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=0.1,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1
    )

    # Train the agent
    model.learn(total_timesteps=5000)

    # Save the model
    model.save("dqn_dynamic_pricing")

    # Access environment instance for results
    env_instance = env.envs[0]

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Time': env_instance.time_stamps,
        'Item_ID': env_instance.item_ids,
        'Price': env_instance.price_history,
        'Profit': env_instance.profit_history
    })

    # Sort and save results
    results_df = results_df.sort_values('Time').reset_index(drop=True)
    results_df.to_csv('dqn_price_time_data.csv', index=False)
    print("\nPrice and time data saved to 'dqn_price_time_data.csv'")

    # Plot price changes
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Time'], results_df['Price'], marker='o', color='blue', label='Price')
    plt.title(f'DQN Price Changes for Item ID {item_id} on {date_of_interest}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dqn_price_changes.png')
    plt.show()

    # Plot cumulative profit
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Time'], results_df['Profit'].cumsum(), marker='o', color='green', label='Cumulative Profit')
    plt.title(f'DQN Cumulative Profit for Item ID {item_id} on {date_of_interest}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Profit')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dqn_cumulative_profit.png')
    plt.show()