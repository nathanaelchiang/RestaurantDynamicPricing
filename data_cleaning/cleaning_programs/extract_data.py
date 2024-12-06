import os
import pandas as pd

def clean_data(df, has_date_column=False):
    """
    Perform basic cleaning on the dataframe:
    - Remove extra spaces from column names.
    - Remove duplicate rows.
    - If 'Date' column exists, ensure it contains only the date (no time).
    """
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # If 'Date' column exists, ensure it contains only date (no time)
    if has_date_column and 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

    return df

def extract_sheets():
    """
    Extracts the 'Items' and 'Orders' sheets from the 'Original_Dataset.xlsx' file,
    cleans the data, and saves them as separate Excel files in the 'datasets' folder.
    """
    try:
        # Locate the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, "../datasets")  # Adjust for parallel folder

        # Construct file paths
        input_file = os.path.join(datasets_dir, "Original_Dataset.xlsx")
        items_output_file = os.path.join(datasets_dir, "Items.xlsx")
        orders_output_file = os.path.join(datasets_dir, "Orders.xlsx")

        # Validate the existence of the input file
        if not os.path.exists(input_file):
            print(f"Error: Original dataset file '{input_file}' not found.")
            return

        # Load datasets
        items_df = pd.read_excel(input_file, sheet_name='Items')
        orders_df = pd.read_excel(input_file, sheet_name='Orders')

        # Validate that the required columns exist in both datasets
        required_columns_orders = ['Item', 'Date']
        required_columns_items = ['Item']

        # Validate columns for Orders
        for column in required_columns_orders:
            if column not in orders_df.columns:
                print(f"Error: The 'Orders' dataset is missing the '{column}' column.")
                return

        # Validate columns for Items
        for column in required_columns_items:
            if column not in items_df.columns:
                print(f"Error: The 'Items' dataset is missing the '{column}' column.")
                return

        # Clean data using the clean_data function
        items_df = clean_data(items_df, has_date_column=False)  # 'Items' doesn't have 'Date' column
        orders_df = clean_data(orders_df, has_date_column=True)  # 'Orders' has 'Date' column

        # Handle missing or invalid data by removing rows with missing 'Date' values (only for Orders)
        orders_df = orders_df.dropna(subset=['Date'])

        # Save the cleaned datasets
        items_df.to_excel(items_output_file, index=False)
        orders_df.to_excel(orders_output_file, index=False)

        print(f"Updated datasets saved successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    extract_sheets()
