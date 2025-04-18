import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    # Download historical stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Reset index to make the date a column
    stock_data.reset_index(inplace=True)
    
    # Rename 'Close' to 'Price' to match the desired format
    stock_data.rename(columns={'Close': 'Price'}, inplace=True)
    
    # Ensure all required columns are present
    required_columns = ['Date', 'Price', 'High', 'Low', 'Open', 'Volume']
    if not all(column in stock_data.columns for column in required_columns):
        raise ValueError("One or more required columns are missing from the downloaded data.")
    
    # Reorder columns to match the desired format
    stock_data = stock_data[required_columns]
    
    return stock_data

# Example usage:
ticker = 'AMZN'  # Amazon stock ticker symbol
start_date = '2015-01-01'  # Start date
end_date = '2025-01-01'    # End date

amazon_stock_prices = get_stock_data(ticker, start_date, end_date)

# Display the first few rows of the data
print(amazon_stock_prices.head())

# Save the data to a CSV file
amazon_stock_prices.to_csv('./dataset/Amazon_large.csv', index=False)