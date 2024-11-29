import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from datetime import datetime, timedelta

from dynamic_pricing_agents.DQN_agent.dqn import DynamicPricingEnvDQN
from dynamic_pricing_agents.PPO_agent.ppo import DynamicPricingEnv


def prepare_daily_data(data_path, item_id, date):
    """
    Prepare data for a specific item and date with fixed DataFrame handling
    """
    # Load the full dataset
    data = pd.read_csv(data_path)

    # Convert date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    mask = (data['Item'] == item_id) & (data['Date'].dt.date == date.date())
    daily_data = data[mask].copy()

    # Create DateTime column before grouping
    daily_data['DateTime'] = pd.to_datetime(daily_data['Date'].astype(str) + ' ' + daily_data['Time'])

    # Group by time periods
    daily_data = daily_data.groupby(['DateTime', 'Item']).agg({
        'Date': 'first',
        'Time': 'first',
        'Price': 'mean',
        'Cost': 'mean',
        'Quantity': 'sum'
    }).reset_index()

    # Sort by DateTime
    daily_data = daily_data.sort_values('DateTime').reset_index(drop=True)

    return daily_data


def generate_comparative_plots(results):
    """
    Generate visualization comparing DQN and PPO performance with teal and coral colors
    """
    teal_color = '#008080'  # Teal
    coral_color = '#FF6B6B'  # Coral/Salmon
    plt.style.use('classic')

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='white')
    fig.patch.set_facecolor('white')

    # 1. Daily Profits Comparison
    ax = axes[0, 0]
    days = range(len(results['dqn']['profits']))
    ax.plot(days, results['dqn']['profits'], label='DQN', marker='o', color=teal_color, linewidth=2)
    ax.plot(days, results['ppo']['profits'], label='PPO', marker='s', color=coral_color, linewidth=2)
    ax.set_title('Daily Profits Comparison', pad=15, fontsize=12, fontweight='bold')
    ax.set_xlabel('Day', fontsize=10)
    ax.set_ylabel('Profit', fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f8f9fa')

    # 2. Price Distribution
    ax = axes[0, 1]
    ax.hist(results['dqn']['prices'], bins=30, alpha=0.6, label='DQN', color=teal_color)
    ax.hist(results['ppo']['prices'], bins=30, alpha=0.6, label='PPO', color=coral_color)
    ax.set_title('Price Distribution', pad=15, fontsize=12, fontweight='bold')
    ax.set_xlabel('Price', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f8f9fa')

    # 3. Price Stability (Rolling Variance)
    ax = axes[1, 0]
    dqn_variance = pd.Series(results['dqn']['prices']).rolling(10).var()
    ppo_variance = pd.Series(results['ppo']['prices']).rolling(10).var()
    ax.plot(dqn_variance, label='DQN', color=teal_color, linewidth=2)
    ax.plot(ppo_variance, label='PPO', color=coral_color, linewidth=2)
    ax.set_title('Price Stability (Rolling Variance)', pad=15, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('Price Variance', fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f8f9fa')

    # 4. Sales vs Price Scatter
    ax = axes[1, 1]
    ax.scatter(results['dqn']['prices'], results['dqn']['sales'],
               alpha=0.6, label='DQN', color=teal_color, s=50)
    ax.scatter(results['ppo']['prices'], results['ppo']['sales'],
               alpha=0.6, label='PPO', color=coral_color, s=50)
    ax.set_title('Sales vs Price', pad=15, fontsize=12, fontweight='bold')
    ax.set_xlabel('Price', fontsize=10)
    ax.set_ylabel('Sales Volume', fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f8f9fa')

    # Remove top and right spines for all plots
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')

    plt.tight_layout(pad=3.0)
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def run_simulation(model, env, num_episodes=1):
    """
    Run a simulation using the trained model
    """
    total_profit = 0
    price_history = []
    sales_history = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_profit = 0

        while not done:
            # Get action from model
            action = model.predict(obs)[0]

            # Take step in environment
            obs, reward, done, info = env.step(action)

            # Record results
            episode_profit += reward
            price_history.append(info['new_price'])
            sales_history.append(info['total_sold'])

        total_profit += episode_profit

    return {
        'total_profit': total_profit / num_episodes,
        'price_history': price_history,
        'sales_history': sales_history
    }


def calculate_performance_metrics(results):
    """
    Calculate key performance metrics for both algorithms
    """
    metrics = {
        'DQN': {
            'average_daily_profit': np.mean(results['dqn']['profits']),
            'profit_std': np.std(results['dqn']['profits']),
            'price_stability': np.std(results['dqn']['prices']),
            'avg_price': np.mean(results['dqn']['prices']),
            'total_sales': sum(results['dqn']['sales'])
        },
        'PPO': {
            'average_daily_profit': np.mean(results['ppo']['profits']),
            'profit_std': np.std(results['ppo']['profits']),
            'price_stability': np.std(results['ppo']['prices']),
            'avg_price': np.mean(results['ppo']['prices']),
            'total_sales': sum(results['ppo']['sales'])
        }
    }

    metrics_df = pd.DataFrame(metrics).round(2)

    # Percent difference column
    metrics_df['PPO_vs_DQN_%'] = ((metrics_df['PPO'] - metrics_df['DQN']) / metrics_df['DQN'] * 100).round(2)

    return metrics_df


def run_comparative_analysis(item_id, date_range, dqn_model_path, ppo_model_path, data_path):
    """Run both algorithms and compare their performance"""
    results = {
        'dqn': {'profits': [], 'prices': [], 'sales': []},
        'ppo': {'profits': [], 'prices': [], 'sales': []}
    }

    print("Loading models...")
    dqn_model = DQN.load(dqn_model_path)
    ppo_model = PPO.load(ppo_model_path)

    # Run simulations for each day
    print(f"\nRunning simulations for {len(date_range)} days...")
    for i, date in enumerate(date_range, 1):
        print(f"Processing day {i}/{len(date_range)}: {date.date()}")

        daily_data = prepare_daily_data(data_path, item_id, date)

        if daily_data.empty:
            print(f"Warning: No data found for item {item_id} on {date.date()}")
            continue

        # Run DQN simulation
        dqn_env = DynamicPricingEnvDQN(daily_data)
        dqn_results = run_simulation(dqn_model, dqn_env)
        results['dqn']['profits'].append(dqn_results['total_profit'])
        results['dqn']['prices'].extend(dqn_results['price_history'])
        results['dqn']['sales'].extend(dqn_results['sales_history'])

        # Run PPO simulation
        ppo_env = DynamicPricingEnv(daily_data)
        ppo_results = run_simulation(ppo_model, ppo_env)
        results['ppo']['profits'].append(ppo_results['total_profit'])
        results['ppo']['prices'].extend(ppo_results['price_history'])
        results['ppo']['sales'].extend(ppo_results['sales_history'])

    print("\nSimulations completed successfully!")
    return results


if __name__ == "__main__":
    # Example date and item
    item_id = 106
    start_date = datetime(2022, 1, 1)
    date_range = [start_date + timedelta(days=x) for x in range(7)]  # One week analysis

    print("\nStarting comparative analysis...")
    results = run_comparative_analysis(
        item_id=item_id,
        date_range=date_range,
        dqn_model_path="DQN_agent/dqn_dynamic_pricing",
        ppo_model_path="PPO_agent/ppo_dynamic_pricing",
        data_path="../data_cleaning/Full_Dataset.csv"
    )

    print("\nGenerating comparison plots...")
    generate_comparative_plots(results)

    print("\nCalculating performance metrics...")
    metrics_df = calculate_performance_metrics(results)
    print("\nPerformance Metrics:")
    print(metrics_df)

    metrics_df.to_csv('performance_metrics.csv')
    print("\nSaved performance metrics to 'performance_metrics.csv'")
