# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 25-03-2026

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# --- Load Dataset ---
file_path = 'heart_rate.csv'
data = pd.read_csv(file_path)

# **********************************************************************
# ADAPTATION: Create a 'datetime' index and set the target column
# since the heart_rate.csv file lacks an explicit date/time column.
# **********************************************************************

# Create synthetic date column (assuming daily frequency) and set index
data['datetime'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
data.set_index('datetime', inplace=True)
data.sort_index(inplace=True)

# Select 'T1' as the target variable (it has no missing values)
target_variable = 'T1'
# Isolate the target variable for the model
data = data[[target_variable]].copy()

print(f"Set target column: {target_variable}")
print("First 5 rows with new index:")
print(data.head())
print("Data shape:", data.shape)

# --- Define ARIMA Modeling Function ---
def arima_model(data, target_variable, order=(1,1,1)):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

    # Fit ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Forecast
    # Use start/end date for forecast index to ensure correct date alignment
    forecast = fitted_model.predict(start=test_data.index[0], end=test_data.index[-1])
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(forecast.index, forecast, label='Forecasted Data', color='red')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title(f'ARIMA ({order}) Forecasting for {target_variable} (Heart Rate Data)')
    plt.legend()
    plt.tight_layout()
    plt.show() # In a notebook environment, this line would be plt.savefig(...)

    return fitted_model, forecast

# --- Run ARIMA Model ---
# Using order=(5,1,0) as in the original request template
model, forecast = arima_model(data, target_variable, order=(5,1,0))
```
### OUTPUT:
<img width="296" height="184" alt="image" src="https://github.com/user-attachments/assets/bc001543-6f53-4709-a833-00669034c76c" />
<img width="938" height="549" alt="image" src="https://github.com/user-attachments/assets/7a6e162b-f146-4351-9298-03ab6d7d388f" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
