import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from datetime import datetime, timedelta

from tqdm import tqdm

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


def calculate_baseline_performance(data_path, item_info, item_id, date_range):
    """Calculate performance metrics if price remained at baseline"""
    baseline_results = {
        'profits': [],
        'prices': [],
        'sales': []
    }

    # Get base price and cost for the item
    item_data = item_info[item_info['Item'] == item_id].iloc[0]
    base_price = item_data['Price']
    cost = item_data['Cost']

    print(f"\nBaseline Analysis for Item {item_id}:")
    print(f"Base Price: ${base_price:.2f}")
    print(f"Cost: ${cost:.2f}")
    print(f"Base Profit Margin: ${base_price - cost:.2f} per unit")

    for date in date_range:
        daily_data = prepare_daily_data(data_path, item_id, date)
        if not daily_data.empty:
            # Calculate daily sales and profit with fixed price
            daily_sales = daily_data['Quantity'].sum()
            daily_profit = (base_price - cost) * daily_sales

            baseline_results['profits'].append(daily_profit)
            baseline_results['prices'].extend([base_price] * len(daily_data))
            baseline_results['sales'].extend(daily_data['Quantity'].tolist())

    return baseline_results


def generate_comparative_plots(results, baseline_results):
    """
    Generate visualization comparing DQN and PPO performance with baseline
    """
    teal_color = '#008080'  # Teal for DQN
    coral_color = '#FF6B6B'  # Coral for PPO
    gray_color = '#808080'  # Gray for baseline
    plt.style.use('classic')

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='white')
    fig.patch.set_facecolor('white')

    # 1. Daily Profits Comparison
    ax = axes[0, 0]
    days = range(len(results['dqn']['profits']))
    ax.plot(days, baseline_results['profits'], label='Baseline', color=gray_color,
            linestyle='--', linewidth=2)
    ax.plot(days, results['dqn']['profits'], label='DQN', marker='o',
            color=teal_color, linewidth=2)
    ax.plot(days, results['ppo']['profits'], label='PPO', marker='s',
            color=coral_color, linewidth=2)
    ax.set_title('Daily Profits Comparison', pad=15, fontsize=12, fontweight='bold')
    ax.set_xlabel('Day', fontsize=10)
    ax.set_ylabel('Profit', fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f8f9fa')

    # 2. Price Distribution
    ax = axes[0, 1]
    ax.hist(baseline_results['prices'], bins=30, alpha=0.6, label='Baseline',
            color=gray_color)
    ax.hist(results['dqn']['prices'], bins=30, alpha=0.6, label='DQN',
            color=teal_color)
    ax.hist(results['ppo']['prices'], bins=30, alpha=0.6, label='PPO',
            color=coral_color)
    ax.set_title('Price Distribution', pad=15, fontsize=12, fontweight='bold')
    ax.set_xlabel('Price', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f8f9fa')

    # 3. Price Stability (Rolling Variance)
    ax = axes[1, 0]
    baseline_variance = pd.Series(baseline_results['prices']).rolling(10).var()
    dqn_variance = pd.Series(results['dqn']['prices']).rolling(10).var()
    ppo_variance = pd.Series(results['ppo']['prices']).rolling(10).var()
    ax.plot(baseline_variance, label='Baseline', color=gray_color,
            linestyle='--', linewidth=2)
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
    ax.scatter(baseline_results['prices'], baseline_results['sales'],
               alpha=0.6, label='Baseline', color=gray_color, s=50)
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
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # New Gymnasium API returns (obs, info)
        else:
            obs = reset_result  # Old Gym API returns just obs

        done = False
        episode_profit = 0

        while not done:
            # Get action from model
            action = model.predict(obs, deterministic=True)[0]

            # Handle both old and new step() return types
            step_result = env.step(action)
            if len(step_result) == 5:  # New Gymnasium API
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # Old Gym API
                obs, reward, done, info = step_result

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


def calculate_performance_metrics(results, baseline_results):
    """
    Calculate key performance metrics including baseline comparison and total profits
    """
    metrics = {
        'Baseline': {
            'total_profit': sum(baseline_results['profits']),
            'average_daily_profit': np.mean(baseline_results['profits']),
            'profit_std': np.std(baseline_results['profits']),
            'price_stability': np.std(baseline_results['prices']),
            'avg_price': np.mean(baseline_results['prices']),
            'total_sales': sum(baseline_results['sales'])
        },
        'DQN': {
            'total_profit': sum(results['dqn']['profits']),
            'average_daily_profit': np.mean(results['dqn']['profits']),
            'profit_std': np.std(results['dqn']['profits']),
            'price_stability': np.std(results['dqn']['prices']),
            'avg_price': np.mean(results['dqn']['prices']),
            'total_sales': sum(results['dqn']['sales'])
        },
        'PPO': {
            'total_profit': sum(results['ppo']['profits']),
            'average_daily_profit': np.mean(results['ppo']['profits']),
            'profit_std': np.std(results['ppo']['profits']),
            'price_stability': np.std(results['ppo']['prices']),
            'avg_price': np.mean(results['ppo']['prices']),
            'total_sales': sum(results['ppo']['sales'])
        }
    }

    metrics_df = pd.DataFrame(metrics).round(2)

    # Calculate improvement percentages over baseline
    for model in ['DQN', 'PPO']:
        metrics_df[f'{model}_vs_Baseline_%'] = (
                (metrics_df[model] - metrics_df['Baseline']) / metrics_df['Baseline'] * 100
        ).round(2)

    # Add PPO vs DQN comparison
    metrics_df['PPO_vs_DQN_%'] = (
            (metrics_df['PPO'] - metrics_df['DQN']) / metrics_df['DQN'] * 100
    ).round(2)

    # Add profit improvements in dollar terms
    metrics_df['DQN_profit_improvement'] = (
            metrics_df['DQN'] - metrics_df['Baseline']
    ).round(2)
    metrics_df['PPO_profit_improvement'] = (
            metrics_df['PPO'] - metrics_df['Baseline']
    ).round(2)
    metrics_df['PPO_vs_DQN_profit_diff'] = (
            metrics_df['PPO'] - metrics_df['DQN']
    ).round(2)

    # Reorder columns for better readability
    column_order = [
        'Baseline', 'DQN', 'PPO',
        'DQN_vs_Baseline_%', 'PPO_vs_Baseline_%', 'PPO_vs_DQN_%',
        'DQN_profit_improvement', 'PPO_profit_improvement', 'PPO_vs_DQN_profit_diff'
    ]
    metrics_df = metrics_df[column_order]

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
        item_info = pd.read_csv('../data_cleaning/ItemPrices_cleaned.csv')
        dqn_env = DynamicPricingEnvDQN(daily_data, item_info)
        dqn_results = run_simulation(dqn_model, dqn_env)
        results['dqn']['profits'].append(dqn_results['total_profit'])
        results['dqn']['prices'].extend(dqn_results['price_history'])
        results['dqn']['sales'].extend(dqn_results['sales_history'])

        # Run PPO simulation
        ppo_env = DynamicPricingEnv(daily_data, item_info)
        ppo_results = run_simulation(ppo_model, ppo_env)
        results['ppo']['profits'].append(ppo_results['total_profit'])
        results['ppo']['prices'].extend(ppo_results['price_history'])
        results['ppo']['sales'].extend(ppo_results['sales_history'])

    print("\nSimulations completed successfully!")
    return results


def run_full_comparative_analysis(
        dqn_model_path,
        ppo_model_path,
        data_path,
        item_prices_path,
        output_path="full_analysis_results.csv",
        start_date=None,
        end_date=None
):
    """
    Run comparative analysis for all items over a full year period.

    Parameters:
    -----------
    dqn_model_path : str
        Path to the trained DQN model
    ppo_model_path : str
        Path to the trained PPO model
    data_path : str
        Path to the full dataset CSV
    item_prices_path : str
        Path to the cleaned item prices CSV
    output_path : str
        Path where to save the results CSV
    start_date : datetime, optional
        Start date for analysis (defaults to Jan 1, 2022)
    end_date : datetime, optional
        End date for analysis (defaults to Dec 31, 2022)

    Returns:
    --------
    tuple
        (item_metrics_df, overall_metrics_dict)
    """
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime(2022, 1, 1)
    if end_date is None:
        end_date = datetime(2022, 12, 31)

    # Generate date range
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    # Load item information
    item_info = pd.read_csv(item_prices_path)

    # Initialize results storage
    all_items_results = []
    overall_metrics = {
        'total_baseline_profit': 0,
        'total_dqn_profit': 0,
        'total_ppo_profit': 0,
        'avg_baseline_price_variance': [],
        'avg_dqn_price_variance': [],
        'avg_ppo_price_variance': []
    }

    # Process each item
    for _, item_row in tqdm(item_info.iterrows(), total=len(item_info), desc="Processing items"):
        item_id = item_row['Item']
        item_name = item_row['Item Name'] if 'Item Name' in item_row else f"Item {item_id}"

        try:
            # Calculate baseline performance
            baseline_results = calculate_baseline_performance(
                data_path,
                item_info,
                item_id,
                date_range
            )

            # Run comparative analysis
            results = run_comparative_analysis(
                item_id=item_id,
                date_range=date_range,
                dqn_model_path=dqn_model_path,
                ppo_model_path=ppo_model_path,
                data_path=data_path
            )

            # Calculate metrics for this item
            baseline_profit = sum(baseline_results['profits'])
            dqn_profit = sum(results['dqn']['profits'])
            ppo_profit = sum(results['ppo']['profits'])

            baseline_price_var = np.var(baseline_results['prices'])
            dqn_price_var = np.var(results['dqn']['prices'])
            ppo_price_var = np.var(results['ppo']['prices'])

            # Calculate profit increases
            dqn_profit_increase = (
                        (dqn_profit - baseline_profit) / baseline_profit * 100) if baseline_profit != 0 else 0
            ppo_profit_increase = (
                        (ppo_profit - baseline_profit) / baseline_profit * 100) if baseline_profit != 0 else 0

            # Store results for this item
            item_results = {
                'item_id': item_id,
                'item_name': item_name,
                'baseline_profit': baseline_profit,
                'dqn_profit': dqn_profit,
                'ppo_profit': ppo_profit,
                'dqn_profit_increase_pct': dqn_profit_increase,
                'ppo_profit_increase_pct': ppo_profit_increase,
                'baseline_price_variance': baseline_price_var,
                'dqn_price_variance': dqn_price_var,
                'ppo_price_variance': ppo_price_var
            }

            all_items_results.append(item_results)

            # Update overall metrics
            overall_metrics['total_baseline_profit'] += baseline_profit
            overall_metrics['total_dqn_profit'] += dqn_profit
            overall_metrics['total_ppo_profit'] += ppo_profit
            overall_metrics['avg_baseline_price_variance'].append(baseline_price_var)
            overall_metrics['avg_dqn_price_variance'].append(dqn_price_var)
            overall_metrics['avg_ppo_price_variance'].append(ppo_price_var)

        except Exception as e:
            print(f"Error processing item {item_id}: {str(e)}")
            continue

    # Create DataFrame with all results
    results_df = pd.DataFrame(all_items_results)

    # Calculate overall averages
    overall_metrics['avg_baseline_price_variance'] = np.mean(overall_metrics['avg_baseline_price_variance'])
    overall_metrics['avg_dqn_price_variance'] = np.mean(overall_metrics['avg_dqn_price_variance'])
    overall_metrics['avg_ppo_price_variance'] = np.mean(overall_metrics['avg_ppo_price_variance'])

    # Add overall profit increases
    overall_metrics['total_dqn_profit_increase_pct'] = (
            (overall_metrics['total_dqn_profit'] - overall_metrics['total_baseline_profit']) /
            overall_metrics['total_baseline_profit'] * 100
    )
    overall_metrics['total_ppo_profit_increase_pct'] = (
            (overall_metrics['total_ppo_profit'] - overall_metrics['total_baseline_profit']) /
            overall_metrics['total_baseline_profit'] * 100
    )

    # Save results to CSV
    results_df.to_csv(output_path, index=False)

    print("\nAnalysis Complete!")
    print(f"\nOverall Metrics:")
    print(f"Total Baseline Profit: ${overall_metrics['total_baseline_profit']:,.2f}")
    print(
        f"Total DQN Profit: ${overall_metrics['total_dqn_profit']:,.2f} ({overall_metrics['total_dqn_profit_increase_pct']:.2f}% increase)")
    print(
        f"Total PPO Profit: ${overall_metrics['total_ppo_profit']:,.2f} ({overall_metrics['total_ppo_profit_increase_pct']:.2f}% increase)")
    print(f"\nAverage Price Variance:")
    print(f"Baseline: {overall_metrics['avg_baseline_price_variance']:.2f}")
    print(f"DQN: {overall_metrics['avg_dqn_price_variance']:.2f}")
    print(f"PPO: {overall_metrics['avg_ppo_price_variance']:.2f}")

    return results_df, overall_metrics


def run_single_item_analysis(item_id):
    # Example date and item
    item_id = 106  # Samosas
    start_date = datetime(2022, 1, 1)
    date_range = [start_date + timedelta(days=x) for x in range(365)]  # One month analysis

    # Load item info for baseline calculation
    item_info = pd.read_csv('../data_cleaning/ItemPrices_cleaned.csv')

    # Calculate baseline performance
    print("\nCalculating baseline performance...")
    baseline_results = calculate_baseline_performance(
        "../data_cleaning/Full_Dataset.csv",
        item_info,
        item_id,
        date_range
    )

    # Run comparative analysis
    print("\nStarting comparative analysis...")
    results = run_comparative_analysis(
        item_id=item_id,
        date_range=date_range,
        dqn_model_path="DQN_agent/dqn_multi_item_pricing",
        ppo_model_path="PPO_agent/ppo_multi_item_pricing",
        data_path="../data_cleaning/Full_Dataset.csv"
    )

    print("\nGenerating comparison plots...")
    generate_comparative_plots(results, baseline_results)

    print("\nCalculating performance metrics...")
    metrics_df = calculate_performance_metrics(results, baseline_results)

    print("\nPerformance Metrics Summary:")
    print(f"\nTotal Profits:")
    print(f"Baseline: ${metrics_df.loc['total_profit', 'Baseline']:,.2f}")
    print(f"DQN: ${metrics_df.loc['total_profit', 'DQN']:,.2f}")
    print(f"PPO: ${metrics_df.loc['total_profit', 'PPO']:,.2f}")

    print(f"\nProfit Improvements:")
    print(f"DQN vs Baseline: ${metrics_df.loc['total_profit', 'DQN_profit_improvement']:,.2f} "
          f"({metrics_df.loc['total_profit', 'DQN_vs_Baseline_%']:,.2f}%)")
    print(f"PPO vs Baseline: ${metrics_df.loc['total_profit', 'PPO_profit_improvement']:,.2f} "
          f"({metrics_df.loc['total_profit', 'PPO_vs_Baseline_%']:,.2f}%)")
    print(f"PPO vs DQN: ${metrics_df.loc['total_profit', 'PPO_vs_DQN_profit_diff']:,.2f} "
          f"({metrics_df.loc['total_profit', 'PPO_vs_DQN_%']:,.2f}%)")

    metrics_df.to_csv('performance_metrics.csv')

if __name__ == "__main__":
    results_df, overall_metrics = run_full_comparative_analysis(
        dqn_model_path="DQN_agent/dqn_multi_item_pricing",
        ppo_model_path="PPO_agent/ppo_multi_item_pricing",
        data_path="../data_cleaning/Full_Dataset.csv",
        item_prices_path="../data_cleaning/ItemPrices_cleaned.csv"
    )