import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yfinance as yf


# Define the base path for the files
base_path = r"C:\Users\rudyk\Downloads\group75Project\Team-75\Data\\"

    # Define file paths
file_paths = {
    'spy_prices': base_path + "SPY Prices.csv",
    'stock_market': base_path + "Stock Market Dataset.csv",
    'us_cpi': base_path + "US CPI - Consumer Price Index.csv",
    'us_fed_funds_rate': base_path + "US FED Funds Rate.csv",
    'us_gdp': base_path + "US GDP.csv",
    'us_job_openings': base_path + "US Job Openings Total Nonfarm.csv",
    'us_pce': base_path + "US PCE - Personal Consumption Expenditure.csv",
    'us_population': base_path + "US Population.csv",
    'us_unemployment_rate': base_path + "US Unemployment Rate.csv"
}

# Read all CSV files into pandas dataframes
dataframes = {}
for name, path in file_paths.items():
    dataframes[name] = pd.read_csv(path)

# Conduct transformations
    
# Remove the 'Unnamed: 0' column from both dataframes
dataframes['spy_prices'].drop(columns=['Unnamed: 0'], inplace=True)
dataframes['stock_market'].drop(columns=['Unnamed: 0'], inplace=True)

# Rename 'Date' column to 'DATE' in spy_prices_df and stock_market_df
dataframes['spy_prices'].rename(columns={'date': 'DATE'}, inplace=True)
dataframes['stock_market'].rename(columns={'Date': 'DATE'}, inplace=True)


# Merge datasets on the 'Date' column
data = pd.merge(dataframes['us_cpi'], dataframes['us_gdp'], on="DATE", how="left")
data = pd.merge(data, dataframes['us_fed_funds_rate'], on="DATE", how="left")
data = pd.merge(data, dataframes['us_job_openings'], on="DATE", how="left")
data = pd.merge(data, dataframes['us_pce'], on="DATE", how="left")
data = pd.merge(data, dataframes['us_population'], on="DATE", how="left")
data = pd.merge(data, dataframes['us_unemployment_rate'], on="DATE", how="left")

# Conduct more transformations
data.rename(columns={'NC000334Q': 'GDP'}, inplace=True)
data['Month'] = pd.to_datetime(data['DATE']).dt.strftime('%Y-%m')
data.columns = ['DATE', 'CPI', 'GDP', 'FEDRATE', 'JO', 'PCE', 'POP', 'UNRATE', 'Month']
data.rename(columns={'Month': 'MONTH'}, inplace=True)


# Define the stock ticker symbol
spy_ticker = "SPY"

# Retrieve stock data
spy_data = yf.download(spy_ticker, start="2013-01-01", end="2024-02-29")
spy_data.reset_index(inplace=True)

spy_data['Month'] = pd.to_datetime(spy_data['Date']).dt.strftime('%Y-%m')

# print(spy_data.head())

# Group by 'Month' and calculate averages
SPY_by_month = spy_data.groupby('Month').agg(
       AVG_PRICE=('Close', 'mean'),  # Replace 'Close' with the correct column name for prices
    AVG_VOLUME=('Volume', 'mean')  # Ensure 'Volume' is the correct column name
).reset_index()

SPY_by_month.rename(columns={'Month': 'MONTH'}, inplace=True)

# Merge data with SPY_by_month on 'MONTH' using a left join
data = data.merge(SPY_by_month, on='MONTH', how='left')

# Calculate the mean for the first and second index positions of 'GDP'
data.loc[0, 'GDP'] = (95.89 + data.loc[2, 'GDP']) / 2
data.loc[1, 'GDP'] = (95.89 + data.loc[2, 'GDP']) / 2

# Set the values for indices 117, 118, 119 to the value at index 116 (assuming zero-based index)
data.loc[117:119, 'GDP'] = data.loc[116, 'GDP']

# Update GDP values in a loop
for i in range(2, 114, 3):  
     if i+3 < len(data):  
        data.loc[i+1, 'GDP'] = np.mean([data.loc[i, 'GDP'], data.loc[i+3, 'GDP']])
        data.loc[i+2, 'GDP'] = np.mean([data.loc[i, 'GDP'], data.loc[i+3, 'GDP']])

# Remove the 121st row 
data = data.drop(index=120)

print(data.head())





