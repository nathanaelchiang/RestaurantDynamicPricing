import pandas as pd

file_path = '../Original_Dataset.xlsx'

orders_df = pd.read_excel(file_path, sheet_name='Orders')
items_df = pd.read_excel(file_path, sheet_name='Items')

# Merging the Orders and Items dataframes on the 'Item' column
merged_df = pd.merge(orders_df, items_df, on='Item', how='inner')

print(merged_df.head())
merged_df.to_csv('Merged_Orders_Items.csv', index=False)
