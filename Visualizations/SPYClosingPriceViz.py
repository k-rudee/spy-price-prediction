import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define the file path
file_path = r"C:\Users\rudyk\Downloads\group75Project\Team-75\Data\SPY Prices.csv"

# Read the CSV file using pandas
spy_prices_df = pd.read_csv(file_path)

# Print Rows to see if read in properly 
print(spy_prices_df.head(), "\n")

# Convert 'date' column to datetime objects for proper plotting
spy_prices_df['date'] = pd.to_datetime(spy_prices_df['date'])

# Set the 'date' column as the index of the dataframe for time series plotting
spy_prices_df.set_index('date', inplace=True)

# Calculate the moving averages
spy_prices_df['MA50'] = spy_prices_df['close'].rolling(window=50).mean()
spy_prices_df['MA200'] = spy_prices_df['close'].rolling(window=200).mean()

# Define the significant events and corresponding colors
event_colors = {
    'End of QE': 'red',
    'Oil Price Crash': 'orange',
    'US Election': 'blue',
    'Tax Cuts and Jobs Act': 'green',
    'Worst Christmas Eve': 'purple',
    'U.S.-China Phase One': 'brown',
    'COVID-19 Pandemic': 'black',
    'Presidential Election': 'grey',
    'Strong Hiring Report': 'cyan',
    'Fed Rate Increase': 'magenta',
    'Inflation Data Reaction': 'yellow'
}

# Define significant dates with actual dates from your dataset
significant_dates = {
    '2014-10-29': 'End of QE',
    '2016-02-11': 'Oil Price Crash',
    '2016-11-08': 'US Election',
    '2017-12-20': 'Tax Cuts and Jobs Act',
    '2018-12-24': 'Worst Christmas Eve',
    '2019-12-13': 'U.S.-China Phase One',
    '2020-03-11': 'COVID-19 Pandemic',
    '2020-11-03': 'Presidential Election',
    '2021-11-05': 'Strong Hiring Report',
    '2022-03-16': 'Fed Rate Increase',
    '2023-03-15': 'Inflation Data Reaction',
    
}

# Create a jitter function to adjust the y-values of the markers
def jitter(values, sd=0.5):
    return values + np.random.normal(0, sd, size=len(values))

# Create the plot
plt.figure(figsize=(14, 7))

# Plot the closing prices and moving averages
plt.plot(spy_prices_df.index, spy_prices_df['close'], label='SPY Close Price', color='navy', linewidth=2)
plt.plot(spy_prices_df.index, spy_prices_df['MA50'], label='50-Day MA', linestyle='--', color='darkorange', linewidth=2)
plt.plot(spy_prices_df.index, spy_prices_df['MA200'], label='200-Day MA', linestyle='--', color='seagreen', linewidth=2)

# Plot the significant events using scatter with jitter on y-values
for date_str, event in significant_dates.items():
    date = pd.to_datetime(date_str)
    close_price = spy_prices_df.at[date, 'close']
    jittered_close_price = jitter(np.array([close_price]))
    plt.scatter(date, jittered_close_price, color=event_colors[event], s=150, label=event, zorder=5)

# Create a custom legend for the events
legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=event, markersize=10, linestyle='None') for event, color in event_colors.items()]
plt.legend(handles=legend_elements, loc='upper left', fontsize=8, title='Significant Events')

# Add the main legend for price and MAs separately
plt.gca().add_artist(plt.legend(loc='lower right'))

# Title, labels and layout
plt.title('SPY Closing Prices with Significant Events and Moving Averages', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price (USD)', fontsize=12)

# Date formatting for the x-axis
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Show and save the plot
plt.tight_layout()
plt.savefig('updated_SPY_Closing_Prices.png', dpi=300)
plt.show()