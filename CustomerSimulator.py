import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import poisson, norm
import random


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

    def setup_time_patterns(self):
        """
        Calculate time-based purchasing patterns from historical data
        """
        # Hour of day patterns (assuming 24-hour format)
        self.hourly_patterns = {
            'breakfast': {'peak': 8, 'std': 1.5, 'weight': 0.3},
            'lunch': {'peak': 12, 'std': 1.5, 'weight': 0.4},
            'dinner': {'peak': 18, 'std': 2, 'weight': 0.3}
        }

        # Day of week patterns (0 = Monday, 6 = Sunday)
        self.weekday_weights = {
            0: 0.8,  # Monday
            1: 0.8,
            2: 0.9,
            3: 1.0,
            4: 1.2,
            5: 1.4,  # Saturday
            6: 1.3  # Sunday
        }

    def setup_price_sensitivity(self):
        """
        Calculate price sensitivity parameters for different customer segments
        """
        self.customer_segments = {
            'price_sensitive': {
                'weight': 0.3,
                'base_elasticity': -2.0,
                'max_price_multiplier': 1.2
            },
            'moderate': {
                'weight': 0.5,
                'base_elasticity': -1.0,
                'max_price_multiplier': 1.5
            },
            'premium': {
                'weight': 0.2,
                'base_elasticity': -0.5,
                'max_price_multiplier': 2.0
            }
        }

    def get_time_multiplier(self, datetime):
        """
        Calculate demand multiplier based on time of day and day of week
        """
        hour = datetime.hour
        weekday = datetime.weekday()

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
        Calculate demand multiplier based on price differences and customer segments.
        Returns 0 if price is 200% or more of base price.
        """
        price_ratio = item_price / base_price

        # Return 0 if price is 200% or more of base price
        if price_ratio >= self.MAX_PRICE_RATIO:
            return 0

        # Weighted average of different customer segment responses
        multiplier = 0
        for segment, params in self.customer_segments.items():
            # Calculate segment-specific response
            if price_ratio > params['max_price_multiplier']:
                segment_multiplier = 0.1  # Minimal demand
            else:
                elasticity_effect = (price_ratio - 1) * params['base_elasticity']
                segment_multiplier = max(0.1, 1 + elasticity_effect)

            multiplier += segment_multiplier * params['weight']

        return multiplier

    def simulate_purchase_decision(self, item_id, price, datetime, quantity_available):
        """
        Simulate whether a purchase occurs based on all factors

        Returns:
        - int: Number of items purchased (0 if no purchase)
        """
        # Get base parameters for the item
        item_data = self.data[self.data['Item'] == item_id]
        if item_data.empty:
            return 0

        base_price = item_data['Price'].mean()
        base_demand = max(0.1, item_data['Count_x'].mean())

        # Check price ceiling first
        if price / base_price >= self.MAX_PRICE_RATIO:
            return 0

        # Calculate various multipliers
        time_mult = self.get_time_multiplier(datetime)
        price_mult = self.get_price_multiplier(price, base_price)

        # If price multiplier is 0, no purchase will occur
        if price_mult == 0:
            return 0

        # Adjust for available quantity
        quantity_mult = min(1.0, quantity_available / base_demand)

        # Calculate final lambda for Poisson distribution
        adjusted_lambda = base_demand * time_mult * price_mult * quantity_mult

        # Generate purchase quantity
        purchase_quantity = min(
            poisson.rvs(adjusted_lambda),
            quantity_available
        )

        return purchase_quantity

    def simulate_day(self, item_id, price, date, initial_quantity):
        """
        Simulate a full day of purchases

        Returns:
        - dict: Summary of the day's transactions
        """
        results = []
        remaining_quantity = initial_quantity

        # Get base price for validation
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

        # Simulate each hour of the day
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
                results.append({
                    'datetime': current_datetime,
                    'quantity': purchase_quantity,
                    'price': price,
                    'revenue': purchase_quantity * price
                })

                remaining_quantity -= purchase_quantity

        return {
            'transactions': results,
            'total_sold': initial_quantity - remaining_quantity,
            'remaining_quantity': remaining_quantity,
            'total_revenue': sum(r['revenue'] for r in results)
        }


# Example usage
if __name__ == "__main__":
    simulator = CustomerSimulator('data_cleaning/Full_Dataset.csv')

    # Test normal price
    test_date = datetime(2024, 10, 30)
    print("\nTesting normal price:")
    simulation_result = simulator.simulate_day(
        item_id=209,
        price=15.99,
        date=test_date,
        initial_quantity=50
    )

    print("\nSimulation Results:")
    print(f"Total items sold: {simulation_result['total_sold']}")
    print(f"Remaining quantity: {simulation_result['remaining_quantity']}")
    print(f"Total revenue: ${simulation_result['total_revenue']:.2f}")

    # Test price above 200% of base price
    print("\nTesting price above 200% of base price:")
    high_price_result = simulator.simulate_day(
        item_id=511,
        price=50.00,  # Assuming this is more than 200% of base price
        date=test_date,
        initial_quantity=50
    )

    print("\nHigh Price Simulation Results:")
    if 'message' in high_price_result:
        print(high_price_result['message'])
    print(f"Total items sold: {high_price_result['total_sold']}")
    print(f"Remaining quantity: {high_price_result['remaining_quantity']}")
    print(f"Total revenue: ${high_price_result['total_revenue']:.2f}")