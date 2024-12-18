
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from cleanData import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
 
# Define function for mad metric
def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Calculate 10-day moving average for AVG_PRICE
data['SMA_10'] = data['AVG_PRICE'].rolling(window=10).mean()

# Add lagged feature for AVG_PRICE (1 day lag)
data['AVG_PRICE_Lag1'] = data['AVG_PRICE'].shift(1)

# Select features and target
features = ['CPI', 'GDP', 'FEDRATE', 'JO', 'PCE', 'POP', 'UNRATE', 'AVG_VOLUME']
target = ['AVG_PRICE']

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[features])
scaled_target = scaler.fit_transform(data[target])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_target, test_size=0.2, random_state=42)

# Reshape input to [samples, time steps, features] for LSTM
X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_target, test_size=0.2, random_state=42)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Adding dropout
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))  # Adding dropout
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32, verbose=2)

# Predicting on the test set
predictions = model.predict(X_test)

# Inverse transform to get actual values
predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate and print metrics
mse = mean_squared_error(y_test_actual, predictions_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
mae = mean_absolute_error(y_test_actual, predictions_actual)
r2 = r2_score(y_test_actual, predictions_actual)
explained_variance = explained_variance_score(y_test_actual, predictions_actual)
mape = mean_absolute_percentage_error(y_test_actual, predictions_actual)
mad = mean_absolute_deviation(y_test_actual, predictions_actual)


# Print the metrics                     
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Explained Variance Score: {explained_variance:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.4f}")
print(f"Mean Absolute Deviation: {mad:.4f}")


# 'predictions_actual' and 'y_test_actual' are the arrays of predictions and actual values, modify dates accordingly
dates = pd.date_range(start='2020-01-01', periods=len(predictions_actual), freq='D')  

# Convert to DataFrame for easier handling
results = pd.DataFrame(data={'Date': dates, 'Actual': y_test_actual.flatten(), 'Predicted': predictions_actual.flatten()})

#plot the results
plt.figure(figsize=(10, 6))
plt.plot(results['Date'], results['Actual'], label='Actual SPY Prices', color='blue')
plt.plot(results['Date'], results['Predicted'], label='Predicted SPY Prices', color='red', linestyle='--')
plt.title('LSTM Model Prediction vs Actual SPY Prices')
plt.xlabel('Date')
plt.ylabel('SPY Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()