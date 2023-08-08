# Forecasting-Indonesian-stock
Forecasting stock price movements in Indonesia, using machine learning, python programming language, and web scrapping method from Yahoo Finance

mport pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

ticker = ['BSSR.JK']
start = datetime.datetime(2019, 1, 1)
end = datetime.datetime(2023, 8, 31)
stock_prices = yf.download(ticker, start=start, end = end, interval='1d').Close

df=pd.DataFrame(stock_prices)
df=df.rename(columns = {"Close": "harga_saham"})
df

df.harga_saham.plot(figsize=(15,5))

df=df.resample('M').mean()
df
df.harga_saham.plot(figsize=(15,5))

df

cols=list()

for i in range(1,0,-1):
  cols.append(df.shift(i))
cols

agg=pd.concat(cols,axis=1)
agg['y']=df['harga_saham']
agg

agg.dropna(inplace=True)
agg

dates = pd.date_range('2019-02-28',periods=len(agg),freq='M')

dates

agg=agg.reset_index()
agg=agg.drop(['Date'],axis=1)
agg

agg['Date']=dates
agg
agg=agg.set_index("Date")

agg

train=agg.head(len(agg)-10)
train

test=agg.tail(11)

cutoff="1990-01-31"
fig, ax = plt.subplots(figsize=(15, 5))
train['harga_saham'].plot(ax=ax, label='Training Set', title='Data Train/Test Split',color="r")
test['harga_saham'].plot(ax=ax, label='Test Set',color="b")
ax.axvline(cutoff, color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

VARIABLE = ['harga_saham']
TARGET = 'y'

X_train = train[VARIABLE]
y_train = train[TARGET]

X_test = test[VARIABLE]
y_test = test[TARGET]

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
model = LinearRegression()
model.fit(X_train, y_train)

model.score(X_train,y_train)

model.score(X_test,y_test)

test['prediction']= model.predict(X_test)

test

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
score = np.sqrt(mean_squared_error(test['y'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')
print("MSE Score on Test set: "+ str(mean_squared_error(test['y'], test['prediction'])))
print("MAE Score on Test set: "+ str(mean_absolute_error(test['y'], test['prediction'])))

df_new = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
df_new

ax = df[['harga_saham']].plot(figsize=(15, 5))
df_new['prediction'].plot(ax=ax, style='-')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
ax.axvline(cutoff, color='black', ls='--')
plt.show()

#Forecast September 2023
forecast=pd.DataFrame([])
forecast['harga_saham']=[3735]
forecast
result= model.predict(forecast)
result

#Forecast oktober 2023
forecast=pd.DataFrame([])
forecast['harga_saham']=[3788.30142561]
forecast
result= model.predict(forecast)
result

