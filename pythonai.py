import pandas as pd
import numpy as np
import plotly.express as px

# CSV 파일 경로 설정
stock_url = "/Users/yunjaehun/desktop/유데미/fi_ai/stocks.csv"

# CSV 파일 읽기
stock_result = pd.read_csv(stock_url)

# 날짜 기준으로 정렬
sorting_result = stock_result.sort_values(by=["Date"])

def interactive_plot(df, title):
    fig = px.line(title = title);
    for i in df.columns[1:]:
        fig.add_scatter(x = df["Date"], y = df[i], name= i );
    fig.show();

interactive_plot(sorting_result,"test");