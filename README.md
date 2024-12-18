# SPY Price Prediction

Project analyzing SPY ETF price movements using machine learning and time series analysis to predict market trends and support investment decisions.

## Overview

This project implements various machine learning and statistical methods to predict SPY ETF prices, including:
- Linear Regression
- Decision Trees
- Random Forests
- Gradient Boosting
- LSTM Neural Networks
- Time Series Analysis (ARIMA, Exponential Smoothing, Prophet)

## Installation

### Requirements
- Python 3.x
- R (for Linear Regression analysis)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Keras
- Statsmodels
- Darts
- Seaborn

## Project Structure

```
├── Code/
│   ├── Linear_Regression.R
│   ├── cleanData.py
│   ├── LSTMmodel.py
│   ├── spy_price_prediction_models.py
│   ├── modelVisualizations.py
│   └── TimeSeries_forecasts_sp500idx.ipynb
├── Data/
│   └── [data files]
└── README.md
```

## Code Description

1. `cleanData.py`: Data preprocessing and cleaning pipeline
   - Merges multiple CSV files containing economic indicators and stock market data
   - Performs data transformations and cleaning
   - Prepares dataset for modeling

2. `LSTMmodel.py`: LSTM neural network implementation
   - Splits data into training/testing sets
   - Implements LSTM architecture
   - Evaluates model performance using various metrics (MSE, RMSE, MAE, R², MAPE)

3. `spy_price_prediction_models.py`: Traditional ML models
   - Implements Decision Tree, Random Forest, and Gradient Boosting
   - Performs hyperparameter tuning
   - Calculates performance metrics

4. `modelVisualizations.py`: Visualization scripts
   - Creates comparison plots
   - Generates feature importance visualizations
   - Produces prediction vs actual price plots

5. `TimeSeries_forecasts_sp500idx.ipynb`: Time series analysis notebook
   - Implements ARIMA models
   - Explores exponential smoothing
   - Utilizes Prophet for forecasting

## Usage

1. Ensure all dependencies are installed
2. Place your data files in the `Data` directory
3. Run the preprocessing script:
```bash
python cleanData.py
```
4. Execute desired model scripts:
```bash
python LSTMmodel.py
python spy_price_prediction_models.py
```
5. Generate visualizations:
```bash
python modelVisualizations.py
```

## Skills Used

1. Machine Learning
2. Time Series Analysis
3. Python/R Programming
4. Financial Data Analysis
5. Statistical Modeling

## Notes

- Update file paths in `cleanData.py` to match your data locations
- Jupyter notebook can be run using:
```bash
jupyter notebook TimeSeries_forecasts_sp500idx.ipynb
```
