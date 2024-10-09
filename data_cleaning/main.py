import pandas as pd

# Load the Excel file
file_path = 'Original_Dataset.xlsx'

# Load the 'Orders' and 'Items' sheets
orders_df = pd.read_excel(file_path, sheet_name='Orders')
items_df = pd.read_excel(file_path, sheet_name='Items')

# Merging the Orders and Items dataframes on the 'Item' column
merged_df = pd.merge(orders_df, items_df, on='Item', how='inner')

# Display the merged dataset
print(merged_df.head())

# Save the merged dataset to a new file if needed
merged_df.to_csv('Merged_Orders_Items.csv', index=False)
