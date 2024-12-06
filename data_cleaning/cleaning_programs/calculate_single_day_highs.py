import pandas as pd
import os

def validate_data(df):
    """
    Validate that the dataframe contains necessary columns and that the 'Count' column has valid numeric values.
    """
    required_columns = ['Item', 'Item Name', 'Count', 'Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Ensure 'Count' column is numeric and non-negative
    df['Count'] = pd.to_numeric(df['Count'], errors='coerce')
    
    # Filter out rows with invalid or negative counts
    invalid_count_rows = df[df['Count'] < 0].shape[0]
    if invalid_count_rows > 0:
        print(f"Warning: {invalid_count_rows} rows with negative or invalid 'Count' have been removed.")
    
    df = df[df['Count'] >= 0]
    
    # Drop rows with missing 'Count' or 'Date'
    invalid_rows = df.isnull().sum().sum()
    if invalid_rows > 0:
        print(f"Warning: {invalid_rows} rows with missing values have been removed.")
    
    df = df.dropna(subset=['Date', 'Count'])
    
    return df

def aggregate_sales(df):
    """
    Group the data by 'Item', 'Item Name', and 'Date' to calculate total sales per day for each item.
    """
    return df.groupby(['Item', 'Item Name', 'Date']).agg({'Count': 'sum'}).reset_index()

def calculate_single_day_highs():
    """
    Calculates the single-day sales highs for items in the Merged_Orders_Items.xlsx file,
    assigns realistic daily starting quantities based on dynamic item popularity, ensures logical consistency, 
    and saves the result in Single_Day_Highs.xlsx.
    """
    try:
        # Locate the script and dataset directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, "../datasets")
        
        # Input and output paths
        input_file = os.path.join(datasets_dir, "Merged_Orders_Items.xlsx")
        output_file = os.path.join(datasets_dir, "Single_Day_Highs.xlsx")
        
        # Validate if the input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Missing input file: {input_file}")

        # Load the merged dataset from the Excel file
        df = pd.read_excel(input_file)

        # Validate data
        df = validate_data(df)

        # Ensure 'Date' column exists and convert to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        # Aggregate daily item sales
        daily_item_sales = aggregate_sales(df)

        # Find the single-day high for each item
        single_day_highs = daily_item_sales.groupby(['Item', 'Item Name']).agg({'Count': 'max'}).reset_index()
        single_day_highs.rename(columns={'Count': 'Single Day High'}, inplace=True)

        # Dynamically assign a buffer for popular items based on total sales quantile
        # Calculate total sales per item across all days
        total_sales_per_item = daily_item_sales.groupby(['Item']).agg({'Count': 'sum'}).reset_index()

        # Define the top percentage (e.g., top 20% of items by total sales)
        top_percentage = 0.2
        sales_threshold = total_sales_per_item['Count'].quantile(1 - top_percentage)  # Top 20%

        # Assign buffer to top-selling items
        def assign_daily_starting_quantity(row):
            # Use the item name buffer if available, otherwise, apply a dynamic buffer based on quantiles
            buffer = 5 if total_sales_per_item.loc[
                total_sales_per_item['Item'] == row['Item'], 'Count'].iloc[0] >= sales_threshold else 2
            return max(10, row['Single Day High'] + buffer)

        single_day_highs['Daily Starting Quantity'] = single_day_highs.apply(assign_daily_starting_quantity, axis=1)

        # Ensure Daily Starting Quantity is at least as large as Single Day High
        single_day_highs['Daily Starting Quantity'] = single_day_highs.apply(
            lambda row: max(row['Daily Starting Quantity'], row['Single Day High']), axis=1
        )

        # Save the single-day highs to an Excel file
        single_day_highs.to_excel(output_file, index=False)
        print(f"Updated dataset saved successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    calculate_single_day_highs()
