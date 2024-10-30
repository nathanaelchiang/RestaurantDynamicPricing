import pandas as pd


def clean_item_numbers(df):
    """
    Clean item numbers by removing any trailing spaces.

    Parameters:
    df (pandas.DataFrame): DataFrame with an 'Item' column containing numbers with spaces

    Returns:
    pandas.DataFrame: DataFrame with cleaned item numbers
    """
    # Strip any whitespace from the Item column
    df['Item'] = df['Item'].str.strip()

    return df


# Example usage:
# Read the CSV
data = pd.read_csv('ItemPrices.csv')

# Clean the item numbers
cleaned_data = clean_item_numbers(data)

# Save the cleaned data back to CSV
cleaned_data.to_csv('ItemPrices_cleaned.csv', index=False)