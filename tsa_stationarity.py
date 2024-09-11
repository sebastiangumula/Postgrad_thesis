# -*- coding: utf-8 -*-
"""
@author: Sebastian Gumula

"""

# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import kpss

df = pd.read_csv("retail_sales_dataset.csv", header=0, index_col=0)

time_series = df[['Date','Total Amount']]

time_series['Date'] = pd.to_datetime(df['Date'])
time_series.set_index('Date', inplace=True)

weekly_sales = time_series.resample('w').sum()

df_values = weekly_sales['Total Amount'].values     

#Perform augmented Dickey-Fuller test
adf_test = adfuller(weekly_sales)

autocorrelation_plot(weekly_sales)
plot_acf(weekly_sales, adjusted=True)
plot_pacf(weekly_sales, method="ols")



#Print ADF results
print("ADF Statistic:" ,adf_test[0])
print("p-value:" ,adf_test[1])
print("Critical values :" ,adf_test[4])

lb_test_stat, lb_p_value = acorr_ljungbox(weekly_sales, lags=1, return_df=False)
print(f"Ljung-Box Test Statistic: {lb_test_stat}")
print(f"P-value: {lb_p_value}")

kpss_stat, kpss_p_value, _, _ = kpss(weekly_sales)

print(f'KPSS Statistic: {kpss_stat}')
print(f'p-value: {kpss_p_value}')

