import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import poisson, norm
import numpy as np


class CustomerSimulator:
    def __init__(self, data_path):
        """
        Initialize the customer simulator with historical data
        """
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.setup_time_patterns()
        self.setup_price_sensitivity()
        self.MAX_PRICE_RATIO = 2.0  # No purchases allowed at 200% or more of base price

        # Initialize memory of recent purchases for momentum
        self.purchase_history = {}

    def setup_time_patterns(self):
        """
        Calculate time-based purchasing patterns from historical data
        """
        # Hour of day patterns (24-hour format) - Adjusted for smoother transitions
        self.hourly_patterns = {
            'breakfast': {'peak': 8, 'std': 2.0, 'weight': 0.3},
            'lunch': {'peak': 12, 'std': 2.0, 'weight': 0.4},
            'dinner': {'peak': 18, 'std': 2.5, 'weight': 0.3}
        }

        # Day of week patterns - More gradual progression
        self.weekday_weights = {
            0: 0.85,  # Monday
            1: 0.90,
            2: 0.95,
            3: 1.0,
            4: 1.1,
            5: 1.2,  # Saturday
            6: 1.15  # Sunday
        }

    def setup_price_sensitivity(self):
        """
        Calculate price sensitivity parameters for different customer segments
        Adjusted for more balanced price sensitivity
        """
        self.customer_segments = {
            'price_sensitive': {
                'weight': 0.25,  # Reduced from 0.3
                'base_elasticity': -1.5,  # Less extreme than -2.0
                'max_price_multiplier': 1.3  # Slightly more tolerant
            },
            'moderate': {
                'weight': 0.5,
                'base_elasticity': -0.8,  # Less sensitive than -1.0
                'max_price_multiplier': 1.6
            },
            'premium': {
                'weight': 0.25,  # Increased from 0.2
                'base_elasticity': -0.3,  # Less sensitive than -0.5
                'max_price_multiplier': 2.0
            }
        }

    def get_time_multiplier(self, datetime_obj):
        """
        Calculate demand multiplier based on time of day and day of week
        """
        hour = datetime_obj.hour
        weekday = datetime_obj.weekday()

        # Calculate time of day effect using mixture of normal distributions
        time_multiplier = 0
        for meal, params in self.hourly_patterns.items():
            time_multiplier += params['weight'] * norm.pdf(
                hour,
                params['peak'],
                params['std']
            )

        # Normalize time multiplier and apply weekday weight
        time_multiplier = time_multiplier * 5  # Scale factor
        weekday_multiplier = self.weekday_weights[weekday]

        return time_multiplier * weekday_multiplier

    def get_price_multiplier(self, item_price, base_price):
        """
        Enhanced price multiplier calculation with diminishing returns
        """
        price_ratio = item_price / base_price

        if price_ratio >= self.MAX_PRICE_RATIO:
            return 0

        multiplier = 0
        for segment, params in self.customer_segments.items():
            # Calculate segment-specific response with diminishing returns
            if price_ratio > params['max_price_multiplier']:
                segment_multiplier = 0.1
            else:
                # Add diminishing returns effect
                elasticity_effect = (price_ratio - 1) * params['base_elasticity']
                # Sigmoid function to create smoother transition
                diminishing_factor = 1 / (1 + np.exp(2 * (price_ratio - params['max_price_multiplier'])))
                segment_multiplier = max(0.1, (1 + elasticity_effect) * diminishing_factor)

            multiplier += segment_multiplier * params['weight']

        # Add small random variation (market noise)
        noise = np.random.normal(1, 0.02)  # 2% standard deviation
        multiplier *= noise

        return max(0.1, multiplier)

    def simulate_purchase_decision(self, item_id, price, datetime_obj, quantity_available):
        """
        Enhanced purchase simulation with momentum effects
        """
        item_data = self.data[self.data['Item'] == item_id]
        if item_data.empty:
            print(f"No historical data found for Item ID {item_id}.")
            return 0

        base_price = item_data['Price'].mean()
        base_demand = max(0.1, item_data['Count_x'].mean())

        if price / base_price >= self.MAX_PRICE_RATIO:
            return 0

        # Calculate various multipliers
        time_mult = self.get_time_multiplier(datetime_obj)
        price_mult = self.get_price_multiplier(price, base_price)

        if price_mult == 0:
            return 0

        # Add momentum effect based on recent purchase history
        momentum_mult = self.calculate_momentum_effect(item_id, datetime_obj)

        # Adjust for available quantity with smoother transition
        quantity_ratio = quantity_available / base_demand
        quantity_mult = 2 / (1 + np.exp(-3 * quantity_ratio)) - 1  # Sigmoid function

        # Calculate final lambda for Poisson distribution
        adjusted_lambda = base_demand * time_mult * price_mult * quantity_mult * momentum_mult

        # Generate purchase quantity with minimum threshold
        purchase_quantity = max(
            1,  # Minimum purchase of 1 if any purchase occurs
            min(
                poisson.rvs(adjusted_lambda),
                quantity_available
            )
        ) if np.random.random() < 0.7 else 0  # 70% chance of purchase if conditions are met

        return purchase_quantity

    def calculate_momentum_effect(self, item_id, datetime_obj):
        """
        Calculate momentum effect based on recent purchase history
        """
        if item_id not in self.purchase_history:
            self.purchase_history[item_id] = []

        # Clean old history (keep last 24 hours)
        self.purchase_history[item_id] = [
            (dt, qty) for dt, qty in self.purchase_history[item_id]
            if (datetime_obj - dt).total_seconds() <= 86400
        ]

        if not self.purchase_history[item_id]:
            return 1.0

        # Calculate recent purchase momentum
        recent_quantities = [qty for _, qty in self.purchase_history[item_id]]
        momentum = sum(recent_quantities) / len(recent_quantities)

        # Normalize momentum effect
        momentum_mult = 1 + 0.1 * np.tanh(momentum - 1)  # Using tanh for smooth bounded effect

        return momentum_mult

    def simulate_day(self, item_id, price, date, initial_quantity):
        """
        Simulate a full day of purchases with enhanced tracking
        """
        results = []
        remaining_quantity = initial_quantity

        # Validate price
        item_data = self.data[self.data['Item'] == item_id]
        if not item_data.empty:
            base_price = item_data['Price'].mean()
            if price / base_price >= self.MAX_PRICE_RATIO:
                return {
                    'transactions': [],
                    'total_sold': 0,
                    'remaining_quantity': initial_quantity,
                    'total_revenue': 0,
                    'message': f'No purchases simulated. Price (${price:.2f}) is {(price / base_price) * 100:.1f}% of base price (${base_price:.2f}), exceeding 200% limit.'
                }

        # Simulate each hour with momentum tracking
        for hour in range(24):
            if remaining_quantity <= 0:
                break

            current_datetime = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)

            purchase_quantity = self.simulate_purchase_decision(
                item_id,
                price,
                current_datetime,
                remaining_quantity
            )

            if purchase_quantity > 0:
                # Record transaction
                results.append({
                    'datetime': current_datetime,
                    'quantity': purchase_quantity,
                    'price': price,
                    'revenue': purchase_quantity * price
                })

                # Update remaining quantity
                remaining_quantity -= purchase_quantity

                # Update purchase history for momentum calculation
                if item_id not in self.purchase_history:
                    self.purchase_history[item_id] = []
                self.purchase_history[item_id].append((current_datetime, purchase_quantity))

        return {
            'transactions': results,
            'total_sold': initial_quantity - remaining_quantity,
            'remaining_quantity': remaining_quantity,
            'total_revenue': sum(r['revenue'] for r in results)
        }