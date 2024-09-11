# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 14:39:57 2024

@author: Sebastian Gumula
"""

from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df = pd.read_csv("retail_sales_dataset.csv", header=0, index_col=0)

time_series = df[['Date','Total Amount']]

time_series['Date'] = pd.to_datetime(df['Date'])
time_series.set_index('Date', inplace=True)

weekly_sales = time_series.resample('w').sum()

result = seasonal_decompose(weekly_sales, model = 'additive', period=7)

# Plot the decomposed components
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.title("Time series decomposition")
plt.plot(weekly_sales['Total Amount'], label='Original')
plt.legend(loc='upper left')
plt.ylabel("Sales")

plt.subplot(412)
plt.plot(result.trend, label='Trend', color='green')
plt.legend(loc='upper left')
plt.ylabel("Sales")

plt.subplot(413)
plt.plot(result.seasonal, label='Seasonal', color= 'orange')
plt.legend(loc='upper left')
plt.ylabel("Sales")

plt.subplot(414)
plt.plot(result.resid, label='Residual', color='magenta')
plt.legend(loc='upper left')
plt.ylabel("Sales")
plt.xlabel("Date")

print(len(weekly_sales))
print(weekly_sales.describe())