import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras

# CSV 파일을 읽어오기
stock_price_df = pd.read_csv("/Users/yunjaehun/desktop/유데미/fi_ai/stock_16/stock.csv")
stock_vol_df = pd.read_csv("/Users/yunjaehun/desktop/유데미/fi_ai/stock_16/stock_volume.csv")

# 정규화
def normalize(df):
    x = df.copy();
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x;

# 데이터 통합
def indivdual_stock(price_df, vol_df, name):
    return pd.DataFrame({"Date" : price_df["Date"],
                         "Close" : price_df[name],
                         "Volume" : vol_df[name]});

price_volume_df = indivdual_stock(stock_price_df, stock_vol_df, "sp500");
print(price_volume_df);
print("------------------------------")
# close, volume 
training_data = price_volume_df.iloc[:, 1:3].values
print(training_data)
print("------------------------------")

# 정규화
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1));
training_set_scaled = sc.fit_transform(training_data);
print(training_set_scaled);
print("------------------------------")

# test data
x = [];
y = [];
for i in range(1, len(price_volume_df)):
    x.append(training_set_scaled[i-1:i, 0]); # 배열의 이전 시점
    y.append(training_set_scaled[i, 0]); # 현재 시점

# 행 열 확인
print(np.shape(x));
print(np.shape(y)); 
print("------------------------------")
X = np.asarray(x);
Y = np.asarray(y);
print("------------------------------")
print(X);
print("------------------------------")
print(Y);
print("------------------------------")
# 70% data추출
split = int(0.7 * len(x));
x_train = X[:split];
y_train = Y[:split];
x_test = X[split:];
y_test = Y[split:];
print("------------------------------")
# 3D 배열로 변환
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_test.shape);
print(x_train.shape);
print("-------------LSTM-------------")
# LSTM 모델 정의
inputs = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences=True)(inputs)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.LSTM(150)(x)
output = keras.layers.Dense(1, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss="mse")
# 모델 요약
model.summary()

# 모델 훈련
history = model.fit(x_train, y_train, epochs = 25, batch_size = 32, validation_split = 0.2);

# 예측, X는 배열의 이전시점
predicted = model.predict(X);

test_predicted = [];
#test_predicted = [i[0] for i in predicted]
for i in predicted:
    test_predicted.append(i[0]);

df_predicted = price_volume_df[1:][["Date"]];
df_predicted["predictions"] = test_predicted;

close = [];
for i in training_set_scaled:
    close.append(i[0]);
df_predicted["Close"] = close[1:];
print(df_predicted);
print("------------------------------")
print(df_predicted);
print("------------------------------")

# 시각화
def interactive_plot(df,title):
    fig = px.line(title=title);
    for i in df.columns[1:]:
        fig.add_scatter(x = df["Date"], y = df[i], name = i);
    fig.show();
interactive_plot(df_predicted, "testML");
