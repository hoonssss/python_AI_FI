import numpy as np
import pandas as pd

# 데이터 불러오기
data_url = pd.read_csv("/Users/yunjaehun/desktop/유데미/fi_ai/stock_16/stock.csv")
df = pd.DataFrame(data_url).sort_values(by="Date")

# 정규화 함수
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i] / x[i][0]
    return x

# 포트폴리오 할당 함수
def portfolio_allocation(df, weights):
    df_portfolio = df.copy()
    df_portfolio = normalize(df_portfolio)
    
    for counter, stock in enumerate(df_portfolio.columns[1:]):
        df_portfolio[stock] = df_portfolio[stock] * weights[counter] * 1_000_000
    
    df_portfolio["portfolio daily worth in $"] = df_portfolio.drop(columns="Date").sum(axis=1)
    
    df_portfolio["return"] = 0.0000
    for i in range(1, len(df)):
        df_portfolio.loc[i, "return"] = ((df_portfolio["portfolio daily worth in $"][i] - df_portfolio["portfolio daily worth in $"][i-1]) / df_portfolio["portfolio daily worth in $"][i-1]) * 100
    
    df_portfolio.loc[0, "return"] = 0
    return df_portfolio

# 가중치 생성 및 정규화
np.random.seed(101)
weights = np.random.random(len(df.columns) - 1)
weights = weights / np.sum(weights)

# 포트폴리오 할당
df_portfolio = portfolio_allocation(df, weights)
print(df_portfolio)
