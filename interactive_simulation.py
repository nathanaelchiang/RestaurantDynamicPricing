import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import poisson, norm
import random
import time

from customer_simulation import CustomerSimulator


class InteractiveCustomerSimulator(CustomerSimulator):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.agent_names = ["Alice", "Bob", "Charlie", "Diana", "Erik"]
        self.setup_customer_personas()

    def setup_customer_personas(self):
        """
        Create different customer personas with varying preferences
        """
        self.customer_personas = {
            'bargain_hunter': {
                'price_sensitivity': 'high',
                'preferred_times': ['breakfast', 'lunch'],
                'weekend_preference': 0.7,
                'purchase_size_mult': 0.8
            },
            'business_professional': {
                'price_sensitivity': 'low',
                'preferred_times': ['lunch', 'dinner'],
                'weekend_preference': 0.3,
                'purchase_size_mult': 1.2
            },
            'casual_shopper': {
                'price_sensitivity': 'medium',
                'preferred_times': ['breakfast', 'dinner'],
                'weekend_preference': 1.2,
                'purchase_size_mult': 1.0
            }
        }

    def generate_customer(self):
        """
        Generate a random customer with a specific persona
        """
        persona_type = random.choice(list(self.customer_personas.keys()))
        persona = self.customer_personas[persona_type]

        return {
            'type': persona_type,
            'persona': persona,
            'preferred_time': random.choice(persona['preferred_times'])
        }

    def simulate_interactive_purchase(self, item_id, price, datetime_obj, quantity_available, agent_name):
        """
        Simulate an interactive purchase with detailed narrative
        """
        customer = self.generate_customer()
        item_data = self.data[self.data['Item'] == item_id]

        if item_data.empty:
            return {
                'success': False,
                'message': f"Item {item_id} not found in historical data."
            }

        base_price = item_data['Price'].mean()
        price_diff_pct = ((price - base_price) / base_price) * 100

        purchase_quantity = self.simulate_purchase_decision(
            item_id,
            price,
            datetime_obj,
            quantity_available
        )

        # Generate narrative
        narrative = self._generate_purchase_narrative(
            customer,
            agent_name,
            item_id,
            price,
            base_price,
            purchase_quantity,
            datetime_obj
        )

        return {
            'success': True,
            'customer_type': customer['type'],
            'purchase_quantity': purchase_quantity,
            'base_price': base_price,
            'current_price': price,
            'price_difference_pct': price_diff_pct,
            'revenue': purchase_quantity * price,
            'narrative': narrative,
            'timestamp': datetime_obj
        }

    def _generate_purchase_narrative(self, customer, agent_name, item_id, price, base_price, quantity, datetime_obj):
        """
        Generate a narrative description of the purchase interaction
        """
        time_of_day = datetime_obj.strftime("%I:%M %p")
        day_of_week = datetime_obj.strftime("%A")

        price_sentiment = "competitive" if price <= base_price else "premium"
        if price >= base_price * 1.5:
            price_sentiment = "expensive"

        narrative_parts = []
        narrative_parts.append(f"[{time_of_day} on {day_of_week}]")
        narrative_parts.append(f"Agent {agent_name} greets a {customer['type'].replace('_', ' ')}.")

        if quantity > 0:
            narrative_parts.append(
                f"Customer orders {quantity} unit{'s' if quantity > 1 else ''} "
                f"at ${price:.2f} each (base price: ${base_price:.2f})."
            )

            price_diff = ((price - base_price) / base_price) * 100
            if abs(price_diff) >= 5:
                narrative_parts.append(
                    f"Price is {'up' if price_diff > 0 else 'down'} "
                    f"{abs(price_diff):.1f}% from base price."
                )
        else:
            narrative_parts.append(
                f"Customer checks the price (${price:.2f}) but decides not to purchase."
            )

        return " ".join(narrative_parts)


def run_interactive_simulation():
    """
    Run the interactive simulation with user input
    """
    simulator = InteractiveCustomerSimulator('data_cleaning/Full_Dataset.csv')

    print("\nWelcome to the Interactive Customer Simulator!")
    print("============================================")

    while True:
        try:
            # Get simulation parameters
            agent_name = random.choice(simulator.agent_names)
            print(f"\nAssigned Agent: {agent_name}")

            item_id = int(input("Enter item ID (or 0 to exit): "))
            if item_id == 0:
                break

            price = float(input("Enter price: $"))

            # Get current date/time or allow user to specify
            use_current_time = input("Use current time? (y/n): ").lower() == 'y'
            if use_current_time:
                current_datetime = datetime.now()
            else:
                date_str = input("Enter date (YYYY-MM-DD): ")
                time_str = input("Enter time (HH:MM): ")
                current_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

            # Simulate purchase
            result = simulator.simulate_interactive_purchase(
                item_id=item_id,
                price=price,
                datetime_obj=current_datetime,
                quantity_available=100,  # Default available quantity
                agent_name=agent_name
            )

            if result['success']:
                print("\nTransaction Details:")
                print("===================")
                print(result['narrative'])
                if result['purchase_quantity'] > 0:
                    print(f"\nRevenue: ${result['revenue']:.2f}")
            else:
                print(f"\nError: {result['message']}")

        except ValueError as e:
            print(f"\nError: Invalid input - {str(e)}")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

        print("\n" + "=" * 50)

        continue_sim = input("\nSimulate another purchase? (y/n): ").lower()
        if continue_sim != 'y':
            break

    print("\nThank you for using the Interactive Customer Simulator!")


if __name__ == "__main__":
    run_interactive_simulation()