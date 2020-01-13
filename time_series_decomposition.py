# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:42:49 2019

@author: pedro.braga

Time Series Decomposition of SQIA3 stock data,
exposing the time series trend, seasonality, and residuals.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm


df = pd.read_csv(r'/home/pedro/Downloads/SQIA3_SA.csv', delimiter=',', header=0)

ts = df['Adj Close']
ts.index = df['Date']
plt.plot(ts)
decomposition = sm.tsa.seasonal_decompose(ts, model='additive', freq=2)

def plot_all(ts, decomposition):
	plt.title('TS Decomposition')
	plt.plot(ts.index, ts, color='blue', label='original time series')
	plt.legend(loc='best')
	plt.plot(ts.index, decomposition.trend, color = 'red', label='trend')
	plt.legend(loc='best')
	plt.plot(ts.index, decomposition.seasonal, color = 'green', label='seasonal')
	plt.legend(loc='best')
	plt.plot(ts.index, decomposition.resid, color = 'yellow', label='residual')
	plt.legend(loc='best')
	plt.show()
plot_all(ts)

def plot_tight(ts, decomposition):
	plt.title('Tight Decomposition')
	plt.subplot(4,1,1)
	plt.plot(ts.index, ts, color = 'blue', label='ts')
	plt.legend(loc='best')
	plt.subplot(4,1,2)
	plt.plot(ts.index, decomposition.trend, color = 'red', label='trend')
	plt.legend(loc='best')
	plt.subplot(4,1,3)
	plt.plot(ts.index, decomposition.seasonal, color = 'green', label='seasonal')
	plt.legend(loc='best')
	plt.subplot(4,1,4)
	plt.plot(ts.index, decomposition.resid, color = 'yellow', label='residual')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.show()
plot_tight(ts)

def plot_trend(ts, decomposition):
	plt.title('TS Decomposition')
	plt.plot(ts.index, ts, color='blue', label='original time series')
	plt.legend(loc='best')
	plt.plot(ts.index, decomposition.trend, color = 'red', label='trend')
	plt.legend(loc='best')
	plt.scatter(ts.index, ts, color='green')
	plt.show()
plot_trend(ts)


