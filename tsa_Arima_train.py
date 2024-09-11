
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from pmdarima import auto_arima
import numpy as np
import warnings
import itertools

warnings.filterwarnings("ignore")

df = pd.read_csv("retail_sales_dataset.csv", header=0, index_col=0)

time_series = df[['Date','Total Amount']]

time_series['Date'] = pd.to_datetime(df['Date'])
time_series.set_index('Date', inplace=True)

weekly_sales = time_series.resample('w').sum()

df_values = weekly_sales['Total Amount'].values     
  
diff = list()

for i in range(1, len(df_values)):
    value = df_values[i] - df_values[i-1]
    diff.append(value)

#Splitting data into train and test sets
size = 39
train, test = df_values[0:size], df_values[size:len(df_values)]

#Grid search algorithm
"""
# Define the parameter combinations for the grid search
p_values = range(0, 10)  # Adjust based on your expectation
d_values = range(0, 10)  # Adjust based on your expectation
q_values = range(0, 10)  # Adjust based on your expectation

# Generate all possible combinations of p, d, and q
combinations = list(itertools.product(p_values, d_values, q_values))

best_rmse = float('inf')
best_mape = float('inf')
best_order = None

# Iterate through all combinations
for order in combinations:
    try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))
        
        # Calculate RMSE for the current combination
        rmse = sqrt(mean_squared_error(test, predictions))
        mape = mean_absolute_percentage_error(test, predictions)
        # Update the best model if the current combination performs better
        if (rmse < best_rmse and mape < best_mape):
            best_rmse = rmse
            best_order = order
            best_mape = mape

    except Exception as e:
        print(f"Error for order {order}: {e}")

print(best_order, best_rmse)
print(best_order, best_mape)
"""

arimamodel= ARIMA(train, order=(6,0,7))
arimamodel_fit = arimamodel.fit()

print(arimamodel_fit.summary())
history = [x for x in train]
predictions = list()


# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(6,0,7))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t] 
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
    
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot forecasts against actual outcomes
plt.plot(test,label ='test')
plt.plot(predictions, label= 'predictions', color='red')

plt.xlabel("Week")
plt.ylabel("Sales")
plt.title("Arima (6, 0, 7) Model Forecasting")
plt.legend(loc='upper left')

mape = mean_absolute_percentage_error(test, predictions)
print(mape)
