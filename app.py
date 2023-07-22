
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import streamlit as st
from keras.models import load_model
import tensorflow.compat.v2 as tf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# Define the stock symbol and timeframe
# symbol = "TSLA"  
start_date = "2010-01-01"
end_date = "2019-12-31"

st.title('Stock trend predictor ')
user_input = st.text_input('Enter the stock ticker','TSLA')
# Fetch the historical stock data
# data = yf.download(symbol, start=start_date, end=end_date)
data = yf.download(user_input, start=start_date, end=end_date)
# data.head()
# Print the retrieved data
# print(data)
# df = data.DataReader('AAPL' , 'yahoo' , start , end)
# df.head()

# describing data
st.subheader('Data from 2010-2019')
st.write(data.describe())

# visualization
st.subheader('Closing price vs time')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)


plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Clossing price vs time with 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(data.Close, 'b')
st.pyplot(fig)

# splitting data into training and testing
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])
# print(data_training.shape)
# print(data_testing.shape)

scaler = MinMaxScaler(feature_range=(0,1))  
data_training_array = scaler.fit_transform(data_training)

# splitting data into x_train and y_train
# we have deleted the training stuff bcs we used the pretrained model
       

# load my model
model = load_model('keras_model.h5')

#  make predictions or the testing part 
# //////////////////////////////////////////
ignore_index=True
past_100_days = data_training.tail(100)
final_data = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_data)
    
x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test) , np.array(y_test)
y_predicted = model.predict(x_test)
sclaer = scaler.scale_

# scaler we won't be the same for every stock
#  so made the scaler at zero i.e. scaler[0]
scale_factor = 1 / scaler.data_min_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Graph
st.subheader('predictions vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

