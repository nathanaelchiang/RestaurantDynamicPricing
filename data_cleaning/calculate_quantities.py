import pandas as pd

# Load the merged dataset
file_path = 'Merged_Orders_Items.csv'
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Step 1: Calculate the single day high for each item
daily_item_sales = df.groupby(['Item', 'Item Name', 'Date']).agg({'Count': 'sum'}).reset_index()
single_day_highs = daily_item_sales.groupby(['Item', 'Item Name']).agg({'Count': 'max'}).reset_index()

# Merge the single day highs back to the main dataframe
df = pd.merge(df, single_day_highs[['Item', 'Item Name', 'Count']], on=['Item', 'Item Name'], how='left')
df.rename(columns={'Count_y': 'Single Day High'}, inplace=True)

# Step 2: Initialize the quantity for each day with the single day high
df['Quantity'] = df['Single Day High']


# Step 3: Subtract 1 from quantity after each order and reset at the start of each day
# We iterate over the dataframe and apply the rule: subtract 1 for each order and reset at the start of a new day
def update_quantity(df):
    df = df.sort_values(by=['Date', 'Time'])  # Ensure the data is sorted by date and time
    current_day = None
    quantities = {}

    for i, row in df.iterrows():
        item = row['Item']
        if row['Date'] != current_day:  # If it's a new day, reset the quantities
            current_day = row['Date']
            quantities = {item: row['Single Day High'] for item in df['Item'].unique()}

        # Update quantity after the order
        df.at[i, 'Quantity'] = quantities[item]
        quantities[item] -= 1  # Subtract 1 after each order

    return df


# Apply the update quantity function to the dataframe
df = update_quantity(df)

# Optionally, save the updated dataframe to a new CSV file
df.to_csv('Updated_Orders_Items_Quantities.csv', index=False)

# Display the first few rows to verify
df.head()
