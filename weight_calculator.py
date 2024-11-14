import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.stats import norm


class WeightCalculator:
    def __init__(self, data_path):
        """
        Initialize with historical sales data
        """
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Hour'] = self.data['Date'].dt.hour
        self.data['Weekday'] = self.data['Date'].dt.dayofweek

    def calculate_time_patterns(self):
        """
        Calculate actual meal time patterns from data with improved curve fitting
        """
        # Group by hour and calculate average sales
        hourly_sales = self.data.groupby('Hour')['Count_x'].mean()

        # Normalize the sales data to help with fitting
        sales_normalized = hourly_sales / hourly_sales.max()

        def triple_normal(x, b_weight, b_mean, b_std,
                          l_weight, l_mean, l_std,
                          d_weight, d_mean, d_std):
            """Modified to ensure weights are positive and sum to reasonable values"""
            return (b_weight * norm.pdf(x, b_mean, b_std) +
                    l_weight * norm.pdf(x, l_mean, l_std) +
                    d_weight * norm.pdf(x, d_mean, d_std))

        # Set up bounds for the parameters
        bounds = ([
                      0.1, 6, 0.5,  # breakfast min (weight, mean, std)
                      0.1, 11, 0.5,  # lunch min
                      0.1, 16, 0.5  # dinner min
                  ], [
                      0.5, 10, 3.0,  # breakfast max
                      0.5, 14, 3.0,  # lunch max
                      0.5, 20, 3.0  # dinner max
                  ])

        try:
            # Fit with better initial parameters and bounds
            hours = np.array(range(24))
            popt, _ = curve_fit(
                triple_normal,
                hours,
                sales_normalized,
                p0=[0.3, 8, 1.5, 0.4, 12, 1.5, 0.3, 18, 2],
                bounds=bounds,
                maxfev=5000,  # Increase max iterations
                method='trf'  # Use trust region reflective algorithm
            )

            # Normalize weights to sum to 1
            total_weight = popt[0] + popt[3] + popt[6]
            meal_patterns = {
                'breakfast': {
                    'weight': popt[0] / total_weight,
                    'peak': popt[1],
                    'std': popt[2]
                },
                'lunch': {
                    'weight': popt[3] / total_weight,
                    'peak': popt[4],
                    'std': popt[5]
                },
                'dinner': {
                    'weight': popt[6] / total_weight,
                    'peak': popt[7],
                    'std': popt[8]
                }
            }

        except RuntimeError:
            # Fallback to simple averaging if curve fitting fails
            print("Warning: Curve fitting failed, using simple averaging method instead.")

            # Define time windows for each meal
            meal_windows = {
                'breakfast': (6, 10),
                'lunch': (11, 14),
                'dinner': (17, 21)
            }

            # Calculate average sales for each window
            meal_averages = {}
            for meal, (start, end) in meal_windows.items():
                meal_sales = hourly_sales[start:end + 1].mean()
                meal_averages[meal] = meal_sales

            # Calculate weights based on proportions
            total_sales = sum(meal_averages.values())
            meal_patterns = {
                meal: {
                    'weight': sales / total_sales,
                    'peak': sum(window) / 2,  # middle of the window
                    'std': 1.5  # reasonable default
                }
                for meal, sales, window in zip(
                    meal_averages.keys(),
                    meal_averages.values(),
                    meal_windows.values()
                )
            }

        return meal_patterns

    def calculate_weekday_weights(self):
        """
        Calculate actual day-of-week patterns from data
        """
        # Group by weekday and calculate average sales
        daily_sales = self.data.groupby('Weekday')['Count_x'].mean()

        # Normalize weights relative to the mean
        weekday_weights = daily_sales / daily_sales.mean()

        return weekday_weights.to_dict()

    def calculate_price_segments(self):
        """
        Calculate price sensitivity segments based on actual purchase patterns
        """
        # Calculate base price for each item
        item_base_prices = self.data.groupby('Item')['Price'].mean()

        # Calculate price ratios for all transactions
        self.data['base_price'] = self.data['Item'].map(item_base_prices)
        self.data['price_ratio'] = self.data['Price'] / self.data['base_price']

        # Get the bin edges and ensure we have at least some variation
        price_ratio_std = self.data['price_ratio'].std()

        if price_ratio_std < 0.01:  # If there's very little variation
            print("Warning: Very little price variation detected. Using default segments.")
            # Create default segments instead of modifying the data
            return {
                'price_sensitive': {
                    'weight': 0.6,
                    'base_elasticity': -1.5,
                    'max_price_multiplier': 1.2
                },
                'moderate': {
                    'weight': 0.3,
                    'base_elasticity': -1.0,
                    'max_price_multiplier': 1.35
                },
                'premium': {
                    'weight': 0.1,
                    'base_elasticity': -0.5,
                    'max_price_multiplier': 1.5
                }
            }

        try:
            # Calculate quartiles
            bins = pd.qcut(self.data['price_ratio'], q=3, retbins=True)[1]
            # Create a categorical variable with explicit string labels
            price_ranges = pd.cut(
                self.data['price_ratio'],
                bins=bins,
                labels=['price_sensitive', 'moderate', 'premium'],
                include_lowest=True
            )

            # Calculate segment weights
            segment_volumes = self.data.groupby(price_ranges, observed=True)['Count_x'].sum()
            segment_weights = segment_volumes / segment_volumes.sum()

            # Calculate elasticities
            def safe_elasticity(group):
                if len(group) < 2:
                    return -1.0
                avg_price_ratio = group['price_ratio'].mean()
                avg_demand = group['Count_x'].mean()
                if avg_price_ratio == 1 or avg_demand == 0:
                    return -1.0
                return -1 * (avg_demand - group['Count_x'].mean()) / (avg_price_ratio - 1)

            elasticities = self.data.groupby(price_ranges, observed=True).apply(safe_elasticity)

            # Create segments dictionary with string keys
            segments = {}
            for i, label in enumerate(['price_sensitive', 'moderate', 'premium']):
                if i < len(segment_weights):
                    segments[label] = {
                        'weight': float(segment_weights.get(label, 1 / 3)),
                        'base_elasticity': float(elasticities.get(label, -1.0)),
                        'max_price_multiplier': float(bins[i + 1]) if i + 1 < len(bins) else 1.5
                    }

        except (ValueError, IndexError) as e:
            print(f"Warning: Error in price segmentation ({str(e)}). Using default segments.")
            segments = {
                'price_sensitive': {
                    'weight': 0.6,
                    'base_elasticity': -1.5,
                    'max_price_multiplier': 1.2
                },
                'moderate': {
                    'weight': 0.3,
                    'base_elasticity': -1.0,
                    'max_price_multiplier': 1.35
                },
                'premium': {
                    'weight': 0.1,
                    'base_elasticity': -0.5,
                    'max_price_multiplier': 1.5
                }
            }

        return segments

    def print_analysis(self):
        """
        Print a comprehensive analysis of the calculated weights
        """
        print("\nCalculating time patterns...")
        meal_patterns = self.calculate_time_patterns()

        print("Calculating weekday patterns...")
        weekday_weights = self.calculate_weekday_weights()

        print("Calculating customer segments...")
        customer_segments = self.calculate_price_segments()

        print("\n=== Time Pattern Analysis ===")
        for meal, params in meal_patterns.items():
            print(f"\n{meal.capitalize()}:")
            print(f"  Weight: {params['weight']:.2f}")
            print(f"  Peak Hour: {params['peak']:.1f}")
            print(f"  Standard Deviation: {params['std']:.1f}")

        print("\n=== Weekday Pattern Analysis ===")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day_num, weight in weekday_weights.items():
            print(f"{days[day_num]}: {weight:.2f}")

        print("\n=== Customer Segment Analysis ===")
        for segment, params in customer_segments.items():
            print(f"\n{segment.replace('_', ' ').capitalize()}:")
            print(f"  Segment Size: {params['weight']:.2f}")
            print(f"  Price Elasticity: {params['base_elasticity']:.2f}")
            print(f"  Max Price Multiplier: {params['max_price_multiplier']:.2f}")


# Example usage
if __name__ == "__main__":
    calculator = WeightCalculator('data_cleaning/Full_Dataset.csv')
    calculator.print_analysis()