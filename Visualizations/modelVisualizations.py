import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cleanData import *
from LSTMmodel import *
from spy_price_prediction_models import y_test, y_pred_tree, y_pred_rf, gb_grid_search, poly_features

# Decision Tree Visualization
# Actual vs. Predicted Prices Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_tree, alpha=0.7, label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Actual')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices (Decision Tree)')
plt.legend()
plt.tight_layout()
plt.show()

# Random Forest Visualization
# Residual Plot
residuals_rf = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals_rf, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot (Random Forest)')
plt.tight_layout()
plt.show()

# Gradient Boosting Visualization
# Feature Importance Plot
feature_importances_gb = gb_grid_search.best_estimator_.feature_importances_
sorted_indices_gb = np.argsort(feature_importances_gb)[::-1]
sorted_features_gb = poly_features[sorted_indices_gb]
sorted_importances_gb = feature_importances_gb[sorted_indices_gb]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features_gb[:10], sorted_importances_gb[:10])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances (Gradient Boosting)')
plt.tight_layout()
plt.show()

# LSTM Model Prediction Visualization
dates = pd.date_range(start='2020-01-01', periods=len(predictions_actual), freq='D')  # Modify according to your dataset's dates

# Convert to DataFrame for easier handling
results = pd.DataFrame(data={'Date': dates, 'Actual': y_test_actual.flatten(), 'Predicted': predictions_actual.flatten()})

# Plot the results
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