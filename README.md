# **Dynamic Pricing Optimization for a Restaurant**

## **Overview**

This project uses advanced reinforcement learning models—Proximal Policy Optimization (PPO) and Deep Q-Learning (DQN)—to dynamically adjust menu prices. The goal is to maximize daily profits by balancing the trade-off between customer demand and profitability. The project simulates customer behavior using historical data and AI-driven dynamic pricing strategies.

## **Features**

- **Reinforcement Learning Models**:
  - **PPO (Proximal Policy Optimization)**: Optimizes pricing with stable adjustments and maximized profits.
  - **DQN (Deep Q-Network)**: Explores more aggressive pricing strategies for short-term profit spikes.
- **Customer Simulation**:

  - Models customer behavior using historical sales data.
  - Simulates three customer personas - price-senstivie, moderate, and premium, with varying levels of price sensitivity.

- **Performance Visualization**:

  - Tracks daily profits, sales patterns, price variance, and sales-price relationships for model evaluation.

- **Dynamic Pricing Strategies**:
  - Applies AI-based pricing to maximize profits while maintaining customer trust.
  - Provides insights into optimal price ranges for menu items.

---

## **Document Overview**

- **arc_elasticity.py**: Analyzes seasonal trends and price elasticity using historical data.
- **distributions.py**: Models statistical demand distributions and simulates customer reactions.
- **weight_calculator.py**: Identifies time-based sales patterns and optimizes weights for pricing strategies.

- **customer_simulation.py**: Creates a simulation environment to model customer behavior and price sensitivity.
- **interactive_simulation.py**: Extends customer simulation with individual customer personas for a realistic environment.

- **analysis.py**: Compares DQN and PPO pricing strategies against a baseline over multiple items and dates.
- **dqn.py**: Trains Deep Q-Learning model to optimize pricing strategies.
- **ppo.py**: Trains Proximal Policy Optimization model to optimize pricing strategies.

- **algorithm_comparison.png**: Visual comparison of baseline pricing vs. DQN and PPO performance.

---

## **Getting Started**

### **Prerequisites**

Ensure you have the following installed:

- Python 3.7+
- pandas
- numpy
- matplotlib
- scipy
- stable-baselines3 (for reinforcement learning)
- gym (for environment simulations)

### **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/ashk0821/RestaurantDynamicPricing.git
   ```

2. Navigate to the project directory:

   ```bash
   cd RestaurantDynamicPricing
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run statistc calculations:

   ```bash
   cd statistic_calculations
   python arc_elasticity.py
   python distributions.py
   python weight_calculator.py
   ```

5. Run the customer simulation:

   ```bash
   cd ../customer_simulation
   python customer_simulation.py
   python interactive_simulation.py
   ```

6. Train and test PPO and DQN models:
   ```bash
   cd ../dynamic_pricing_agents/PPO_agent
   python ppo.py
   ```
   ```bash
   cd ../DQN_agent
   python dqn.py
   ```
7. Compare model performance:
   ```bash
   cd ../analysis
   python analysis.py
   ```

---

## **Using the Models**

1. **Simulate Customer Behavior**:

   - Run `customer_simulation.py` to simulate demand patterns.
   - Use `interactive_simulation.py` to explore how different personas respond to pricing changes.

2. **Optimize Pricing**:

   - Train the PPO and DQN models to identify optimal price ranges.
   - Evaluate model performance with visual metrics (e.g., profits, price variance).

3. **Deploy Dynamic Pricing**:
   - Apply insights from the trained models to set prices dynamically while maintaining customer trust.

---

## **Results**

- **PPO**: Offers stable and consistent pricing adjustments, balancing profitability and customer retention. It is ideal for higher margin items.
- **DQN**: Explores aggressive pricing strategies but may sacrifice stability for short-term gains. It is ideal for low-margin items.

---

## **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request for new features or optimizations.

---

## **Acknowledgments**

--[insert]
