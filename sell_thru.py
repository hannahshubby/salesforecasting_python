import pandas as pd
from prophet import Prophet
import sys

in_prod = sys.argv[1]
out_filenm=sys.argv[2]
cut_date=sys.argv[3]
periods_cnt=int(sys.argv[4])

df_sell_thru = pd.read_csv("/home/jupyter/sell_thru_df.csv", sep=",")
df_sell_thru

df_sell_thru["yymmdd"]=df_sell_thru['yymm'].apply(str)+"01"

df_sell_thru_prod=df_sell_thru.query("Product==@in_prod")
df_sell_thru_prod.head()

if cut_date != "":
  df_sell_thru_prod=df_sell_thru_prod.query("yymmdd>=@cut_date")

df_sell_thru_prod.rename(columns = {'Quantity' : 'y'}, inplace = True)
df_sell_thru_prod.rename(columns = {'yymmdd' : 'ds'}, inplace = True)


model = Prophet()
model.fit(df_sell_thru_prod[["ds", "y"]])
future = model.make_future_dataframe(periods=periods_cnt, freq='MS')
forecast = model.predict(future)

forecast["year"]=pd.DatetimeIndex(forecast['ds']).year#.astype(str)
forecast["month"]=pd.DatetimeIndex(forecast['ds']).month #.astype(str)
df_sell_thru_prod["year"]=pd.DatetimeIndex(df_sell_thru_prod['ds']).year#.astype(str)
df_sell_thru_prod["month"]=pd.DatetimeIndex(df_sell_thru_prod['ds']).month#.astype(str)

df_full_forecast = pd.merge(df_sell_thru_prod, forecast, left_on=['year', 'month'], right_on=['year','month'], how='outer')
df_full_forecast.to_csv('/home/jupyter/'+out_filenm, sep=',', na_rep='NaN', mode='w',)
