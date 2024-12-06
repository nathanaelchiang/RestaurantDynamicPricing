import pandas as pd
import os

# Locate the script and datasets directory
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, "../datasets")

# File paths
orders_items_file = os.path.join(datasets_dir, "Merged_Orders_Items.xlsx")
single_day_highs_file = os.path.join(datasets_dir, "Single_Day_Highs.xlsx")
output_file = os.path.join(datasets_dir, "Full_Dataset.xlsx")

# Validate if input files exist
if not os.path.exists(orders_items_file):
    raise FileNotFoundError(f"Orders and Items file not found: {orders_items_file}")
if not os.path.exists(single_day_highs_file):
    raise FileNotFoundError(f"Single Day Highs file not found: {single_day_highs_file}")

# Load the datasets
df = pd.read_excel(orders_items_file)
single_day_highs = pd.read_excel(single_day_highs_file)

# Format and validate 'Date' and 'Count' columns
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
df = df.dropna(subset=['Date'])  # Drop invalid dates
df['Count'] = pd.to_numeric(df['Count'], errors='coerce')
df = df.dropna(subset=['Count'])  # Drop invalid counts

# Merge the single day highs into the main dataframe
df = pd.merge(df, single_day_highs[['Item', 'Item Name', 'Single Day High', 'Daily Starting Quantity']], 
              on=['Item', 'Item Name'], how='left')

# Initialize the quantity for each day with the daily starting quantity
df['Available Stock'] = df['Daily Starting Quantity']

# Update quantities dynamically based on orders
def update_quantity(df):
    df = df.sort_values(by=['Date', 'Time'])  # Sort by date and time
    current_day = None
    quantities = {}

    for i, row in df.iterrows():
        item = row['Item']
        if row['Date'] != current_day:
            current_day = row['Date']
            quantities = {item: row['Daily Starting Quantity'] for item in df['Item'].unique()}
        
        # Update quantity: subtract the count from current stock
        quantities[item] -= row['Count']
        
        # Ensure that stock doesn't go below zero
        if quantities[item] < 0:
            quantities[item] = 0
        
        # Ensure stock doesn't exceed the daily starting quantity
        quantities[item] = min(quantities[item], row['Daily Starting Quantity'])

        # Update the Available Stock column
        df.at[i, 'Available Stock'] = quantities[item]
    
    return df

# Apply the update quantity function
df = update_quantity(df)

# Save the updated dataset
df.to_excel(output_file, index=False)
print(f"Updated dataset saved successfully.")
