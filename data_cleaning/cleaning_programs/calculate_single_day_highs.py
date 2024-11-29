import pandas as pd

file_path = '../Merged_Orders_Items.csv'
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Group the data by 'Item', 'Item Name', and 'Date', then sum the 'Count' to get total sales per day for each item
daily_item_sales = df.groupby(['Item', 'Item Name', 'Date']).agg({'Count': 'sum'}).reset_index()

# Find the single day high for each item by finding the max 'Count' for each 'Item' and 'Item Name'
single_day_highs = daily_item_sales.groupby(['Item', 'Item Name']).agg({'Count': 'max'}).reset_index()

print(single_day_highs)
single_day_highs.to_csv('Single_Day_Highs.csv', index=False)

