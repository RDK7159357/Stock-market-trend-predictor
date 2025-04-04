import yfinance as yf

# Download historical data
stock_df = yf.download("AMZN", start='2020-01-01', end='2025-1-1')

# Save to CSV
stock_df.to_csv("Amazon_company_stock_data.csv")

# Print head
print(stock_df.head())
