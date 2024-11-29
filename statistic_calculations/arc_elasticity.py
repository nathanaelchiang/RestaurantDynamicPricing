import pandas as pd
import numpy as np
from scipy.stats import poisson

df = pd.read_csv('../data_cleaning/Full_Dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])


# Create a function to determine the season
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


# Add season column
df['Season'] = df['Date'].apply(get_season)

# Group by Item, Season, and Date to get daily sales
daily_sales = df.groupby(['Item', 'Season', 'Date'])['Count_x'].sum().reset_index()

# Calculate average daily sales for each item in each season
avg_sales = daily_sales.groupby(['Item', 'Season'])['Count_x'].mean().reset_index()
avg_sales.columns = ['Item', 'Season', 'Avg_Daily_Sales']

# Calculate average price for each item in each season
avg_price = df.groupby(['Item', 'Season'])['Price'].mean().reset_index()
avg_price.columns = ['Item', 'Season', 'Avg_Price']

# Merge average sales and average price back to the original dataframe
df = pd.merge(df, avg_sales, on=['Item', 'Season'], how='left')
df = pd.merge(df, avg_price, on=['Item', 'Season'], how='left')

# Replace zero Avg_Daily_Sales with a small positive number to avoid division by zero
df['Avg_Daily_Sales'] = df['Avg_Daily_Sales'].replace(0, 0.1)

# Replace zero Avg_Price with a small positive number to avoid division by zero
df['Avg_Price'] = df['Avg_Price'].replace(0, 0.1)

# Calculate Arc Price Elasticity
df['Price_Elasticity'] = -1 * (
        ((df['Count_x'] - df['Avg_Daily_Sales']) / ((df['Count_x'] + df['Avg_Daily_Sales']) / 2))
        *
        ((df['Price'] - df['Avg_Price']) / ((df['Price'] + df['Avg_Price']) / 2))
)

# Handle cases where Avg_Daily_Sales or Avg_Price might lead to undefined elasticity
df['Price_Elasticity'] = df['Price_Elasticity'].replace([np.inf, -np.inf], 0).fillna(0)


# Function to simulate demand
def simulate_demand(row, num_simulations=1000):
    lambda_param = max(0.1, row['Avg_Daily_Sales'])

    # Adjust lambda based on price elasticity and quantity left
    # Using arc elasticity, so we don't adjust based on Avg_Daily_Sales again
    price_factor = max(0.1, 1 + (row['Price_Elasticity'] * (row['Price'] - row['Avg_Price']) / row['Avg_Price']))
    quantity_factor = min(1, max(0.1, row['Quantity'] / lambda_param))

    adjusted_lambda = max(0.1, lambda_param * price_factor * quantity_factor)

    # Generate Poisson distribution
    simulated_demand = poisson.rvs(adjusted_lambda, size=num_simulations)

    return simulated_demand.mean()


# Apply simulation to each row
df['Simulated_Demand'] = df.apply(simulate_demand, axis=1)

print(df[['Date', 'Item', 'Season', 'Price', 'Quantity', 'Avg_Daily_Sales', 'Simulated_Demand']])

# Average Simulated Demand by Season
season_demand = df.groupby('Season')['Simulated_Demand'].mean()
print("\nAverage Simulated Demand by Season:")
print(season_demand)

# Top 5 Items by Average Simulated Demand
item_demand = df.groupby('Item')['Simulated_Demand'].mean().sort_values(ascending=False)
print("\nTop 5 Items by Average Simulated Demand:")
print(item_demand.head())
