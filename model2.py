
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from stock_data import stockData
import sys
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')



#Verileri stockData fonksiyonundan çekiyoruz
data_set = np.array(stockData("XU100"))
prices = data_set.reshape(-1, 1)

X = prices[:-1]  
y = prices[1:]   




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_train_flat = X_train.reshape(X_train.shape[0], -1)
x_test_flat = X_test.reshape(X_test.shape[0], -1)

y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_test_flat = y_test.reshape(y_test.shape[0], -1)




scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(x_train_flat)
X_test_scaled = scaler.transform(x_test_flat)


activation_function = "relu"

model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(x_train_flat.shape[1],1)))
model.add(keras.layers.Dense(64,activation=activation_function))
model.add(keras.layers.Dense(64,activation=activation_function))
model.add(keras.layers.Dense(32,activation=activation_function))
model.add(keras.layers.Dense(32,activation=activation_function))
model.add(keras.layers.Dense(1,activation=activation_function))





model.compile(optimizer='adam', loss='mean_absolute_error')


model.fit(X_train_scaled, y_train_flat, epochs=50, batch_size=32, validation_split=0.1)

loss = model.evaluate(X_test_scaled, y_test_flat)
print("Test Loss:", loss)

print(model.summary())

y_pred = model.predict(X_test_scaled)

difference = 0
for i in range(len(y_pred)):
    fark = (abs(y_pred[i]-y_test_flat[i])/y_test_flat[i])
    difference += fark[0]*100
    

print("Test Loss:", loss)
print("accuracy: ",100-(difference/len(y_pred)))





backtest = 50

pred_price = scaler.transform(prices[-(backtest-1):])

time_range = range(len(data_set[-(backtest-1):]))

y_pred = model.predict(pred_price)

plt.figure(figsize=(10, 6))

plt.plot(time_range, data_set[-(backtest-1):], color='blue', label='Gerçek Fiyatlar')

plt.plot(time_range, y_pred, color='green', label='Tahmin Edilen Fiyatlar')

plt.title('Gerçek ve Tahmin Edilen Fiyatlar')
plt.xlabel('Zaman')
plt.ylabel('Fiyat')
plt.legend()
plt.grid(True)
plt.show()



