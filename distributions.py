import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime


class CustomerAgent:
    def __init__(self, data_path):
        """
        Initializes the CustomerAgent by loading and preprocessing the dataset.

        Parameters:
        - data_path (str): Path to the CSV dataset.
        """
        self.data = self.load_and_preprocess_data(data_path)
        self.calculate_price_elasticity()
        self.simulate_demand()

    def load_and_preprocess_data(self, data_path):
        """
        Loads the dataset and performs initial preprocessing.

        Parameters:
        - data_path (str): Path to the CSV dataset.

        Returns:
        - pd.DataFrame: Preprocessed DataFrame.
        """
        # Load the dataset
        df = pd.read_csv(data_path)

        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Add 'Season' column
        df['Season'] = df['Date'].apply(self.get_season)

        # Group by Item, Season, and Date to get daily sales
        daily_sales = df.groupby(['Item', 'Season', 'Date'])['Count_x'].sum().reset_index()

        # Calculate average daily sales for each item in each season
        avg_sales = daily_sales.groupby(['Item', 'Season'])['Count_x'].mean().reset_index()
        avg_sales.columns = ['Item', 'Season', 'Avg_Daily_Sales']

        # Merge average sales back to the original dataframe
        df = pd.merge(df, avg_sales, on=['Item', 'Season'], how='left')

        # Replace zero Avg_Daily_Sales with a small positive number to avoid division by zero
        df['Avg_Daily_Sales'] = df['Avg_Daily_Sales'].replace(0, 0.1)

        return df

    @staticmethod
    def get_season(date):
        """
        Determines the season for a given date.

        Parameters:
        - date (pd.Timestamp): The date to determine the season for.

        Returns:
        - str: Season name ('Winter', 'Spring', 'Summer', 'Fall').
        """
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    def calculate_price_elasticity(self):
        """
        Calculates a simplified price elasticity for each item.
        """
        # Ensure 'Price' and 'Count_x' are numeric
        self.data['Price'] = pd.to_numeric(self.data['Price'], errors='coerce')
        self.data['Count_x'] = pd.to_numeric(self.data['Count_x'], errors='coerce')

        # Calculate price elasticity
        self.data['Price_Elasticity'] = -1 * (
                (self.data['Count_x'] / self.data['Avg_Daily_Sales']) *
                (self.data['Price'] / self.data['Avg_Daily_Sales'])
        )

        # Handle infinite or NaN values
        self.data['Price_Elasticity'].replace([np.inf, -np.inf], 0, inplace=True)
        self.data['Price_Elasticity'].fillna(0, inplace=True)

    def simulate_demand(self, num_simulations=1000):
        """
        Simulates customer demand using a Poisson distribution.

        Parameters:
        - num_simulations (int): Number of simulations to run for averaging.
        """

        def _simulate(row):
            # Base lambda is the average daily sales
            lambda_param = max(0.1, row['Avg_Daily_Sales'])

            # Adjust lambda based on price elasticity and current price
            price_factor = max(
                0.1,
                1 + (row['Price_Elasticity'] * (row['Price'] - row['Avg_Daily_Sales']) / row['Avg_Daily_Sales'])
            )

            # Adjust based on quantity left (assuming 'Quantity' represents inventory)
            quantity_factor = min(
                1,
                max(0.1, row['Quantity'] / lambda_param)
            )

            # Adjusted lambda for Poisson
            adjusted_lambda = max(0.1, lambda_param * price_factor * quantity_factor)

            # Simulate demand
            simulated_demand = poisson.rvs(adjusted_lambda, size=num_simulations)
            return simulated_demand.mean()

        # Apply simulation to each row
        self.data['Simulated_Demand'] = self.data.apply(_simulate, axis=1)

    def get_simulated_data(self):
        """
        Retrieves the DataFrame with simulated demand.

        Returns:
        - pd.DataFrame: DataFrame containing simulated demand.
        """
        return self.data[['Date', 'Item', 'Season', 'Price', 'Quantity', 'Avg_Daily_Sales', 'Simulated_Demand']]

    def analyze_demand(self):
        """
        Provides analysis of the simulated demand.

        Returns:
        - dict: Average demand by season and top 5 items by average demand.
        """
        season_demand = self.data.groupby('Season')['Simulated_Demand'].mean().to_dict()
        item_demand = self.data.groupby('Item')['Simulated_Demand'].mean().sort_values(ascending=False).head(
            5).to_dict()
        return {
            'Average_Simulated_Demand_by_Season': season_demand,
            'Top_5_Items_by_Average_Simulated_Demand': item_demand
        }

    def get_simulated_demand(self, item, season):
        """
        Retrieves the simulated demand for a specific item and season.

        Parameters:
        - item (int or str): The Item identifier.
        - season (str): The season ('Winter', 'Spring', 'Summer', 'Fall').

        Returns:
        - float: Average simulated demand for the specified item and season.
        """
        # Validate season input
        valid_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        if season not in valid_seasons:
            raise ValueError(f"Invalid season '{season}'. Valid options are: {valid_seasons}")

        # Filter data for the specified item and season
        filtered_data = self.data[(self.data['Item'] == item) & (self.data['Season'] == season)]

        if filtered_data.empty:
            raise ValueError(f"No data found for Item {item} in {season} season.")

        # Calculate average simulated demand
        avg_simulated_demand = filtered_data['Simulated_Demand'].mean()
        return avg_simulated_demand

    def get_simulated_demand_distribution(self, item, season, num_simulations=1000):
        """
        Retrieves the simulated demand distribution for a specific item and season.

        Parameters:
        - item (int or str): The Item identifier.
        - season (str): The season ('Winter', 'Spring', 'Summer', 'Fall').
        - num_simulations (int): Number of simulations per data row.

        Returns:
        - np.ndarray: Array of simulated demand values.
        """
        # Validate season input
        valid_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        if season not in valid_seasons:
            raise ValueError(f"Invalid season '{season}'. Valid options are: {valid_seasons}")

        # Filter data for the specified item and season
        filtered_data = self.data[(self.data['Item'] == item) & (self.data['Season'] == season)]

        if filtered_data.empty:
            raise ValueError(f"No data found for Item {item} in {season} season.")

        # Aggregate simulated demand
        simulated_demands = filtered_data['Simulated_Demand'].values
        return simulated_demands


# Usage Example
if __name__ == "__main__":
    # Initialize the CustomerAgent with the path to your dataset
    agent = CustomerAgent(data_path='data_cleaning/Full_Dataset.csv')

    # Retrieve and display simulated demand data
    simulated_data = agent.get_simulated_data()
    print("Sample Simulated Demand Data:")
    print(simulated_data.head())

    # Analyze and display overall demand insights
    demand_analysis = agent.analyze_demand()
    print("\nAverage Simulated Demand by Season:")
    for season, demand in demand_analysis['Average_Simulated_Demand_by_Season'].items():
        print(f"{season}: {demand:.2f}")

    print("\nTop 5 Items by Average Simulated Demand:")
    for item, demand in demand_analysis['Top_5_Items_by_Average_Simulated_Demand'].items():
        print(f"Item {item}: {demand:.2f}")

    # Calculate simulated demand for a specific item and season
    specific_item = 511  # Example Item ID
    specific_season = 'Winter'  # Example Season
    try:
        demand = agent.get_simulated_demand(item=specific_item, season=specific_season)
        print(f"\nSimulated Demand for Item {specific_item} in {specific_season}: {demand:.2f} units/day")
    except ValueError as e:
        print(e)

    # (Optional) Retrieve the simulated demand distribution for further analysis
    try:
        demand_distribution = agent.get_simulated_demand_distribution(item=specific_item, season=specific_season)
        print(f"\nDemand Distribution for Item {specific_item} in {specific_season}:")
        print(demand_distribution)
    except ValueError as e:
        print(e)