import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# CSV 파일 경로
stock_url = "/Users/yunjaehun/desktop/유데미/fi_ai/stocks.csv"

# CSV 파일 읽기
stock_result = pd.read_csv(stock_url)

# 날짜별로 정렬
sorting_result = stock_result.sort_values(by=["Date"])

def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:  # 첫 번째 열은 Date이므로 제외
        for j in range(1, len(df)):
            df_daily_return.loc[j, i] = ((df.loc[j, i] - df.loc[j-1, i]) / df.loc[j-1, i]) * 100
        df_daily_return.loc[0, i] = 0  # 첫 번째 행의 값은 0으로 설정
    return df_daily_return 

# 일일 수익률 계산
stock_daily_return = daily_return(sorting_result);

#plt
stock_daily_return.hist(figsize=(10,10), bins=40);
plt.show();

df_hist = stock_daily_return.copy();
df_hist = df_hist.drop(columns=["Date"]);
data = [];
for i in df_hist.columns:
    data.append(stock_daily_return[i].values);

print(data);

#plotly
fig = ff.create_distplot(data, df_hist.columns);
fig.show();