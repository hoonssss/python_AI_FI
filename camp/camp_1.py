import numpy as np
import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go


read_csv = "/Users/yunjaehun/desktop/유데미/fi_ai/stock_16/stock.csv"

parsing = pd.read_csv(read_csv);
df_data = parsing.sort_values("Date");

# 정규화
def normalize(df):
    x = df.copy();
    for i in x.columns[1:]: #data 제외
        x[i] = x[i]/x[i][0]
    return x;

# 시각화
def interactive_plot(df, input_title):
    fig = px.line(title = input_title);
    for i in df.columns[1:]:
        fig.add_scatter(x = df["Date"], y = df[i], name = i);
    fig.show();

# interactive_plot(df_data,"test");
# interactive_plot(normalize(df_data),"test");

# 일 수익률
def daliy_return(df):
    df_daily_return = df.copy();
    for i in df_daily_return.columns[1:]:
        df_daily_return[i] = df[i].pct_change() * 100
    df_daily_return.iloc[0, 1:] = 0;
    return df_daily_return;

stock_daily_return = daliy_return(df_data);
print(stock_daily_return);
print("---------------------------------");

# 각 기업별 평균 수익률 계산
print(stock_daily_return.iloc[:, 1:].mean())
print("---------------------------------");

# 기업 별 일 수익률
print(stock_daily_return[["AAPL", "sp500"]]);
print("---------------------------------");
# plot
beta, alpha = np.polyfit(stock_daily_return["sp500"], stock_daily_return["AAPL"], 1);
print("beta {}, alpha {}".format(beta,alpha));
stock_daily_return.plot(kind = "scatter", x = "sp500", y = "AAPL");
plt.plot(stock_daily_return["sp500"], beta * stock_daily_return["sp500"] + alpha, '-', color = 'r');
plt.show();
