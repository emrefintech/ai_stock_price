from stock_data import stockData
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt



def MAPE(real, predict):
   
    real, predict = np.array(real), np.array(predict)
    return np.mean(np.abs((real - predict) / real)) * 100




data_set = np.array(stockData("XU100"))
prices = data_set.reshape(-1, 1)



X = prices[:-1]  # Bugünün fiyatı
y = prices[1:]   # Yarınki fiyatı

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Ortalama Kare Hata (MSE):", mse)

mape = MAPE(y_test,y_pred)

print("Ortalama Mutlak Yuzde Hata (MAPE):",mape)

#########################################################

price = prices[-1].reshape(-1,1)
next_pred = model.predict(price)
print("Bir sonraki mumun tahmin edilen fiyati : ", next_pred[0][0])
print("En son alinan fiyat verisi : ", prices[-1][0])

##########################################################

#ÇİZİM

backtest = 50


time_range = range(len(data_set[-(backtest-1):]))

y_pred = model.predict(prices[-(backtest-1):])

plt.figure(figsize=(10, 6))

plt.plot(time_range, data_set[-(backtest-1):], color='blue', label='Gerçek Fiyatlar')

plt.plot(time_range, y_pred, color='green', label='Tahmin Edilen Fiyatlar')

plt.title('Gerçek ve Tahmin Edilen Fiyatlar')
plt.xlabel('Zaman')
plt.ylabel('Fiyat')
plt.legend()
plt.grid(True)
plt.show()





