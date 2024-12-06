import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_full_dataset(input_file, output_file):
    """
    Preprocesses the Full_Dataset.xlsx for PPO training:
    - Removes duplicate columns.
    - Normalizes numerical columns.
    - One-hot encodes Category and Sub Category.
    - Adds Avg_Daily_Sales and Season columns based on Date.
    
    Args:
    - input_file (str): Path to the input dataset.
    - output_file (str): Path to save the processed dataset.
    """
    try:
        # Step 1: Load the dataset
        df = pd.read_excel(input_file)
        
        # Ensure no duplicate columns from the start
        df = df.loc[:, ~df.columns.duplicated()]

        # Step 2: Remove time from 'Date' column, keeping only the date part
        df['Date'] = pd.to_datetime(df['Date']).dt.date  # This keeps only the date (no time)

        # Step 3: Add season column based on 'Date'
        def get_season(date):
            """
            Returns the season based on the given date.
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
        
        # Add season column to the dataset
        df['Season'] = df['Date'].apply(get_season)

        # Step 4: Calculate Avg_Daily_Sales
        # Group by Item, Season, and Date to get daily sales count
        daily_sales = df.groupby(['Item', 'Season', 'Date'])['Available Stock'].sum().reset_index()

        # Calculate the average daily sales for each item in each season
        avg_sales = daily_sales.groupby(['Item', 'Season'])['Available Stock'].mean().reset_index()
        avg_sales.columns = ['Item', 'Season', 'Avg_Daily_Sales']

        # Merge avg_sales back into the original dataframe
        df = pd.merge(df, avg_sales, on=['Item', 'Season'], how='left')

        # Step 5: Feature Engineering
        df['Sales'] = df['Sales Price'] * df['Available Stock']
        df['Profit'] = (df['Sales Price'] - df['Cost Price']) * df['Available Stock']
        df['Profit Margin'] = df['Profit'] / df['Sales Price']
        df['Profit Momentum'] = df.groupby('Item')['Profit'].transform(lambda x: x.diff().fillna(0))

        # Step 6: Normalize numerical columns (overwrite original columns)
        numerical_cols = ['Sales Price', 'Cost Price', 'Profit', 'Profit Momentum', 'Sales', 'Avg_Daily_Sales']
        scaler = StandardScaler()

        # Normalize and update the original columns with the normalized values
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Step 7: One-hot encode Category and Sub Category, then drop original columns
        one_hot_encoded = pd.get_dummies(df[['Category', 'Sub Category']], drop_first=False)
        df = pd.concat([df, one_hot_encoded], axis=1)
        
        # Drop the original 'Category' and 'Sub Category' columns
        df.drop(columns=['Category', 'Sub Category'], inplace=False)

        # Step 8: Save the processed dataset
        df.to_excel(output_file, index=False)
        print(f"Processed dataset saved successfully.")

    except Exception as e:
        print(f"Error during preprocessing: {e}")

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))  # This line is valid only in a Python script
datasets_dir = os.path.join(script_dir, "../datasets")

input_file = os.path.join(datasets_dir, 'Full_Dataset.xlsx')
output_file = os.path.join(datasets_dir, 'Training_Model_Dataset.xlsx')

# Preprocess dataset
preprocess_full_dataset(input_file, output_file)
