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

# 날짜 기준으로 정렬
stock_price_df = stock_price_df.sort_values("Date")
stock_vol_df = stock_vol_df.sort_values("Date")

# null check
print(stock_price_df.isnull().sum());
print(stock_vol_df.isnull().sum());
print("------------------------------")

# df info
print(stock_price_df.info());
print(stock_vol_df.info());
print("------------------------------")

# 데이터 통계
print(stock_price_df.describe());
print(stock_vol_df.describe());

# 정규화
def normalize(df):
    x = df.copy();
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x;

# plotly 시각화
def interactive_plot(df,title):
    fig = px.line(title=title);
    for i in df.columns[1:]:
        fig.add_scatter(x = df["Date"], y = df[i], name = i);
    fig.show();

nor_stock_df = normalize(stock_price_df);
nor_stock_vol_df = normalize(stock_vol_df);

print(normalize(nor_stock_df));
print(normalize(nor_stock_vol_df));
#interactive_plot(nor_stock_vol_df,"test");

# 데이터 통합
def indivdual_stock(price_df, vol_df, name):
    return pd.DataFrame({"Date" : price_df["Date"],
                         "Close" : price_df[name],
                         "Volume" : vol_df[name]});

price_volume_df = indivdual_stock(stock_price_df, stock_vol_df, "sp500");
print(price_volume_df);

# 주가 예측
def traing_window(data):
    n = 1
    data["Target"] = data["Close"].shift(-n);
    return data;

price_volume_target_df = traing_window(price_volume_df);
price_volume_target_df = price_volume_target_df[:-1] # Target Col 마지막 Data 제외
#price_volume_target_df["Target"].iloc[-1] = 0; Target Col 마지막 Data 0 
print(price_volume_target_df);

from sklearn.preprocessing import MinMaxScaler
# 0과 1사이로 정규화
sc = MinMaxScaler(feature_range = (0,1));
price_volume_target_scaled_df = sc.fit_transform(price_volume_target_df.drop(columns=["Date"]));
print(price_volume_target_scaled_df);
print(price_volume_target_scaled_df.shape);

# 2col까지
x = price_volume_target_scaled_df[: , :2];
# 3col만
y = price_volume_target_scaled_df[: , 2:];
print(x)
print(y)

# 65% data만 훈련
split = int(0.65 * len(x));
print(split);
x_train = x[:split];
y_train = y[:split];
x_test = x[split:];
y_test = y[split:];
print(x_train.shape);
print(y_train.shape);
print(x_test.shape);
print(y_test.shape);

def show_plot(data, title):
    plt.figure(figsize=(13,5));
    plt.plot(data, linewidth = 3);
    plt.title(title);
    plt.grid();
    plt.show();

show_plot(x_train,"x_train");
show_plot(x_test,"x_test");
