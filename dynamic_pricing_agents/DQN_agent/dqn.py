import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn
from customer_simulation.customer_simulation import CustomerSimulator


class DynamicPricingEnvDQN(gym.Env):
    def __init__(self, data, item_info, n_discrete_actions=41):
        super(DynamicPricingEnvDQN, self).__init__()

        # Store both transaction data and item information
        self.data = data.reset_index(drop=True)
        self.item_info = item_info
        self.current_step = 0
        self.max_steps = len(self.data) - 1

        # Initialize the CustomerSimulator
        self.customer_simulator = CustomerSimulator('../data_cleaning/Full_Dataset.csv')

        # Define discrete action space
        self.n_discrete_actions = n_discrete_actions
        self.action_space = spaces.Discrete(n_discrete_actions)
        self.price_adjustment_step = 0.2 / (n_discrete_actions - 1)  # Range from -10% to +10%

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            # [price, quantity, profit_momentum, hour_of_day, day_of_week, base_price, cost, category_encoding]
            dtype=np.float32
        )

        # Initialize history
        self.profit_history = {}  # Dictionary to track profit history per item
        self.price_history = {}  # Dictionary to track price history per item
        self.momentum_window = 3

    def _convert_discrete_to_continuous(self, discrete_action):
        """Convert discrete action index to continuous price adjustment percentage"""
        return -0.1 + (discrete_action * self.price_adjustment_step)

    def calculate_profit_momentum(self, item_id):
        """Calculate profit momentum based on recent history for specific item"""
        if item_id not in self.profit_history or len(self.profit_history[item_id]) < 2:
            return 0.0

        recent_profits = self.profit_history[item_id][-self.momentum_window:]
        if len(recent_profits) < 2:
            return 0.0

        return np.mean(np.diff(recent_profits))

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.current_step = 0
        current_item = self.data.loc[self.current_step]['Item']
        self.previous_price = self.item_info.loc[self.item_info['Item'] == current_item, 'Price'].values[0]

        # Reset histories for all items
        self.profit_history = {item: [] for item in self.item_info['Item']}
        self.price_history = {item: [] for item in self.item_info['Item']}

        obs = self._get_observation()
        return obs, {}

    def _get_category_encoding(self, category):
        """Convert category to numerical encoding"""
        category_map = {
            'Appetizers': 0,
            'Main Courses': 1,
            'Sides': 2,
            'Desserts': 3,
            'Beverages': 4
        }
        return category_map.get(category, -1)

    def _get_observation(self):
        current_data = self.data.loc[self.current_step]
        current_datetime = pd.to_datetime(f"{current_data['Date']} {current_data['Time']}")
        current_item = current_data['Item']

        # Get item information
        item_data = self.item_info[self.item_info['Item'] == current_item].iloc[0]

        obs = np.array([
            self.previous_price,
            current_data['Quantity'],
            self.calculate_profit_momentum(current_item),
            current_datetime.hour / 24.0,  # Normalized hour
            current_datetime.dayofweek / 6.0,  # Normalized day of week
            item_data['Price'],  # Base price
            item_data['Cost'],  # Cost
            self._get_category_encoding(item_data['Category']) / 4.0  # Normalized category
        ], dtype=np.float32)
        return obs

    def calculate_reward(self, profit, price_change, item_id):
        """Enhanced reward function with category-specific considerations"""
        # Get item category
        category = self.item_info.loc[self.item_info['Item'] == item_id, 'Category'].values[0]

        # Base reward is profit
        reward = profit

        # Add stability bonus (varies by category)
        stability_weights = {
            'Appetizers': 0.1,
            'Main Courses': 0.15,
            'Sides': 0.05,
            'Desserts': 0.1,
            'Beverages': 0.05
        }
        stability_weight = stability_weights.get(category, 0.1)
        stability_bonus = stability_weight * profit * (1.0 - abs(price_change))

        # Add momentum bonus
        momentum = self.calculate_profit_momentum(item_id)
        momentum_bonus = 0.1 * profit * (1.0 if momentum > 0 else -0.5)

        return reward + stability_bonus + momentum_bonus

    def step(self, action):
        # Convert discrete action to continuous price change
        price_change_percentage = self._convert_discrete_to_continuous(action)

        # Get current data point
        current_data = self.data.loc[self.current_step]
        item_id = current_data['Item']

        # Get item information
        item_info = self.item_info[self.item_info['Item'] == item_id].iloc[0]
        base_price = item_info['Price']
        cost = item_info['Cost']

        date = current_data['Date']
        time = current_data['Time']
        initial_quantity = current_data['Quantity']

        # Apply price adjustment
        new_price = self.previous_price * (1 + price_change_percentage)
        noise = np.random.normal(0, 0.001)
        new_price *= (1 + noise)

        # Ensure new price stays within reasonable bounds
        new_price = min(new_price, base_price * self.customer_simulator.MAX_PRICE_RATIO)
        new_price = max(new_price, base_price * 0.5)

        # Simulate customer behavior
        date_time_str = f"{date} {time}"
        date_time_obj = pd.to_datetime(date_time_str)

        simulation_result = self.customer_simulator.simulate_day(
            item_id=item_id,
            price=new_price,
            date=date_time_obj,
            initial_quantity=initial_quantity
        )

        # Calculate results
        total_sold = simulation_result['total_sold']
        total_revenue = simulation_result['total_revenue']
        remaining_quantity = simulation_result['remaining_quantity']
        profit = (new_price - cost) * total_sold

        # Calculate reward
        reward = self.calculate_reward(profit, price_change_percentage, item_id)

        # Update histories
        if item_id not in self.profit_history:
            self.profit_history[item_id] = []
            self.price_history[item_id] = []

        self.profit_history[item_id].append(profit)
        self.price_history[item_id].append(new_price)
        self.previous_price = new_price

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Get next observation
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)

        info = {
            'item_id': item_id,
            'new_price': new_price,
            'total_sold': total_sold,
            'remaining_quantity': remaining_quantity,
            'total_revenue': total_revenue,
            'profit': profit,
            'price_change': price_change_percentage
        }

        return obs, reward, done, False, info


if __name__ == "__main__":
    # Load datasets
    data = pd.read_csv('../../data_cleaning/Full_Dataset.csv')
    item_info = pd.read_csv('../../data_cleaning/ItemPrices_cleaned.csv')

    # Sort data chronologically
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data = data.sort_values('DateTime').reset_index(drop=True)

    # Initialize environment
    env = DynamicPricingEnvDQN(data, item_info)
    env = DummyVecEnv([lambda: env])

    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=5e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.05,
        gamma=0.99,
        train_freq=8,
        gradient_steps=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs={
            "net_arch": [512, 256, 128],
            "activation_fn": nn.ReLU
        },
        verbose=1
    )

    # Train the agent
    print("Starting training...")
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps)

    model.save("dqn_multi_item_pricing")

    print("Training and evaluation completed")