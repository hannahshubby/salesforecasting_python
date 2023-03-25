import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np


def MAPE(y_test, y_pred):
	return np.mean(np.abs((y_test - y_pred) / y_test)) * 100 

in_prod = "B115"
out_filenm="" # "sell_thru_202302201717.csv"
cut_date = "20210501" #"20210501"

df_sell_thru = pd.read_csv("sell_thru_df.csv", sep=",")
df_sell_thru


df_sell_thru["yymmdd"]=df_sell_thru['yymm'].apply(str)+"01"

df_sell_thru_prod=df_sell_thru.query("Product==@in_prod")

if cut_date != "":
  df_sell_thru_prod=df_sell_thru_prod.query("yymmdd>=@cut_date")

df_sell_thru_prod.head()

df_sell_thru_prod.rename(columns = {'Quantity' : 'sales'}, inplace = True)
df_sell_thru_prod.rename(columns = {'yymmdd' : 'date'}, inplace = True)
df_sell_thru_prod['date'] = pd.to_datetime(df_sell_thru_prod['date'])

df = df_sell_thru_prod.copy()
df = df[["date", "sales"]]


plt.plot(df['sales'])
plt.show()

def adf_test(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

adf_test(df['sales'])
#diff = df.copy()
diff = df.diff().dropna()
adf_test(diff['sales'])

model = ARIMA(np.asarray(diff['sales']), order=(1,1,0))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=12)
forecast = pd.DataFrame(forecast, index=pd.date_range(start='2023-02-01', periods=12, freq='MS'))
forecast=forecast.reset_index()
forecast.columns = ['date', 'sales']
forecast


plt.plot(diff['sales'].to_list(), label='Original')
plt.plot(forecast['sales'].to_list(), label='ARIMA Forecast')
plt.legend()
plt.show()

df_prophet = df.copy()
df_prophet = df_prophet[['date', 'sales']]
df_prophet.columns = ['ds', 'y']
m = Prophet(seasonality_mode='multiplicative')
m.fit(df_prophet)

future = m.make_future_dataframe(periods=12, freq='MS')
forecast_prophet = m.predict(future)
plt.plot(df_prophet['ds'], df_prophet['y'], label='Original')
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Prophet Forecast')
plt.legend()
plt.show()


plt.plot(diff['sales'], label='Original')
plt.plot(forecast['sales'], label='ARIMA Forecast')
plt.plot(forecast_prophet['yhat'], label='Prophet Forecast')
plt.legend()
plt.show()


arima_mape =MAPE( df['sales'].iloc[-12:], forecast['sales'].tail(12) )
prophet_mape =MAPE( df['sales'].iloc[-12:], forecast_prophet['yhat'].tail(12) )

arima_rmse = np.sqrt(mean_squared_error(df['sales'].iloc[-12:], forecast['sales'].tail(12)))
prophet_rmse = np.sqrt(mean_squared_error(df['sales'].iloc[-12:], forecast_prophet['yhat'].tail(12)))
print("ARIMA RMSE:", arima_rmse)
print("Prophet RMSE:", prophet_rmse)

#Prophet 모델의 RMSE 가 더 작아 좀 더 정확한 예측임.
