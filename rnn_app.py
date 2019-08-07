# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:20:15 2019

@author: Eimantas
"""
# Ikelemios reikalingos bibliotekos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

# nuskaitomi mokymosi duomenys
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

# Sutvarkomi duomenys RNN tinklui
sc =  MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Sukuriama duomenu struktura su 60 laiko zingsiu ir vienu isejimu
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# pertavrkymas ( suteikiama dar viena dimensija)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# sukuriamas RNN tinklas
regressor = Sequential()

#Sukuriama pirma LSTM perdanga ir neuronu atjungimo reguliavimas 
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Sukuriama antra LSTM perdanga ir neuronu atjungimo reguliavimas
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Sukuriama trecia LSTM perdanga ir neuronu atjungimo reguliavimas
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Sukuriama ketvirta LSTM perdanga ir neuronu atjungimo reguliavimas
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Pridedama isejimo perdanga
regressor.add(Dense(units = 1))


# sukompiliuojamas RNN tinklas
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# RNN tinklui duodamas mokymosi paketas
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# isaugojamas apmokytas RNN tinklas
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")

# Nuskaitomas tikri duomenys
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Nuspejami duomenys
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Gautas rezultatas atvaizduojamas grafiniu budu
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
