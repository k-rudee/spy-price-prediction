import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from cleanData import *


# Define features and target
features = ['CPI', 'GDP', 'FEDRATE', 'JO', 'PCE', 'POP', 'UNRATE', 'AVG_VOLUME']
target = 'AVG_PRICE'

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(data[features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, data[target], test_size=0.2, random_state=42)

# Hyperparameter Tuning: Reduce the range of hyperparameters
params = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Cross-Validation Strategy: Reduce the number of folds
cv = 5

# Decision Tree
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

# Random Forest
rf_regressor = RandomForestRegressor(random_state=42, n_estimators=100)
rf_grid_search = GridSearchCV(rf_regressor, params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

# Gradient Boosting
gb_regressor = GradientBoostingRegressor(random_state=42, n_estimators=100)
gb_grid_search = GridSearchCV(gb_regressor, params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the models
grid_search.fit(X_train, y_train)
rf_grid_search.fit(X_train, y_train)
gb_grid_search.fit(X_train, y_train)

# Best estimators found by GridSearchCV
best_tree = grid_search.best_estimator_
best_rf = rf_grid_search.best_estimator_
best_gb = gb_grid_search.best_estimator_

# Feature Importance Analysis: Analyze feature importances
feature_importances = best_tree.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

# Get feature names for polynomial features
poly_features = poly.get_feature_names_out(features)
sorted_features = poly_features[sorted_indices]

sorted_importances = feature_importances[sorted_indices]

sorted_importances = feature_importances[sorted_indices]

# Performance Evaluation Metrics: Calculate additional evaluation metrics
y_pred_tree = best_tree.predict(X_test)
y_pred_rf = best_rf.predict(X_test)
y_pred_gb = best_gb.predict(X_test)

# Define calculate_mape function
def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Define calculate_theils_u function
def calculate_theils_u(y_true, y_pred):
    """Calculate Theil's U statistic."""
    num = np.sqrt(np.mean((y_pred - y_true) ** 2))
    den = (np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2)))
    return num / den

# Evaluate
metrics = {
    'Decision Tree': {'mse': mean_squared_error(y_test, y_pred_tree),
                      'r2': r2_score(y_test, y_pred_tree),
                      'mad': mean_absolute_error(y_test, y_pred_tree),
                      'mape': calculate_mape(y_test, y_pred_tree),
                      'theils_u': calculate_theils_u(y_test, y_pred_tree)},
    'Random Forest': {'mse': mean_squared_error(y_test, y_pred_rf),
                      'r2': r2_score(y_test, y_pred_rf),
                      'mad': mean_absolute_error(y_test, y_pred_rf),
                      'mape': calculate_mape(y_test, y_pred_rf),
                      'theils_u': calculate_theils_u(y_test, y_pred_rf)},
    'Gradient Boosting': {'mse': mean_squared_error(y_test, y_pred_gb),
                          'r2': r2_score(y_test, y_pred_gb),
                          'mad': mean_absolute_error(y_test, y_pred_gb),
                          'mape': calculate_mape(y_test, y_pred_gb),
                          'theils_u': calculate_theils_u(y_test, y_pred_gb)}
}

# Print results
for model, metrics_dict in metrics.items():
    print(f"{model}:")
    print(f"  Mean Squared Error: {metrics_dict['mse']:.2f}")
    print(f"  R² Score: {metrics_dict['r2']:.2f}")
    print(f"  Mean Absolute Deviation: {metrics_dict['mad']:.2f}")
    print(f"  Mean Absolute Percentage Error: {metrics_dict['mape']:.2f}%")
    print(f"  Theil's Inequality Coefficient (Theil's U): {metrics_dict['theils_u']:.2f}")

# Model Interpretability: Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:10], sorted_importances[:10])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.show()
