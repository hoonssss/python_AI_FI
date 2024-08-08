import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


# CSV 파일 경로
stock_url = "/Users/yunjaehun/desktop/유데미/fi_ai/stocks.csv"

# CSV 파일 읽기
stock_result = pd.read_csv(stock_url)

# 날짜별로 정렬
sorting_result = stock_result.sort_values(by=["Date"])

def daily_return(df):
    df_daily_return = df.copy()
    # 열이 아니라 행을 기준으로 연산
    for i in df.columns[1:]:  # 첫 번째 열은 Date이므로 제외
        for j in range(1, len(df)):
            df_daily_return.loc[j, i] = ((df.loc[j, i] - df.loc[j-1, i]) / df.loc[j-1, i]) * 100
        df_daily_return.loc[0, i] = 0  # 첫 번째 행의 값은 0으로 설정
    return df_daily_return 

# 일일 수익률 계산
stock_daily_return = daily_return(sorting_result)

# 결과 출력
print(stock_daily_return)

#matplotlib
def show_plot(df, title):
    df.plot(x = "Date", figsize=(15,7), linewidth = 3, title = title);
    plt.grid();
    plt.show();

show_plot(stock_daily_return, "test")

#plotly
def interactive_plot(df, title):
    fig = px.line(title = title);
    for i in df.columns[1:]:
        fig.add_scatter(x = df["Date"], y = df[i], name= i );
    fig.show();

interactive_plot(stock_daily_return,"test");
