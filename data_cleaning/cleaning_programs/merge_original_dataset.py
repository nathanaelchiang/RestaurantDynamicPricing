import os
import pandas as pd

def merge_datasets():
    """
    Merges Orders and Items datasets located in the 'datasets' folder relative to the 'preprocessing' folder,
    and saves the merged dataset as an Excel file in the 'datasets' folder.
    """
    try:
        # Locate the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, "../datasets")  # Adjust for parallel folder

        # Construct file paths
        orders_path = os.path.join(datasets_dir, "Orders.xlsx")
        items_path = os.path.join(datasets_dir, "Items.xlsx")
        output_file = os.path.join(datasets_dir, "Merged_Orders_Items.xlsx")

        # Validate the existence of the datasets folder and input files
        if not os.path.exists(orders_path) or not os.path.exists(items_path):
            raise FileNotFoundError("Required dataset files not found. Ensure 'Orders.xlsx' and 'Items.xlsx' are present in the 'datasets' folder.")

        # Load datasets
        orders_df = pd.read_excel(orders_path)
        items_df = pd.read_excel(items_path)

        # Ensure 'Date' column contains only date and not time (only for Orders)
        if 'Date' in orders_df.columns:
            orders_df['Date'] = pd.to_datetime(orders_df['Date'], errors='coerce').dt.date

        # Merge datasets on the 'Item' column
        merged_df = pd.merge(orders_df, items_df, on='Item', how='inner')

        # Save the merged dataset
        merged_df.to_excel(output_file, index=False)
        print(f"Updated dataset saved successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    merge_datasets()
