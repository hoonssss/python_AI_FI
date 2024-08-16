import pandas as pd
import numpy as np

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
    
    df_portfolio["return"] = df_portfolio["portfolio daily worth in $"].pct_change() * 100
    
    df_portfolio["return"].iloc[0] = 0  # 첫 번째 날의 수익률을 0으로 설정
    
    return df_portfolio

# 가중치 생성 및 정규화
np.random.seed(101)
weights = np.random.random(len(df.columns) - 1)
weights = weights / np.sum(weights)

# 포트폴리오 할당
df_portfolio = portfolio_allocation(df, weights)

# 누적 수익률 계산
result_aapl = ((df_portfolio["AAPL"].iloc[-1] / df_portfolio["AAPL"].iloc[0]) - 1) * 100
result_sp500 = ((df_portfolio["sp500"].iloc[-1] / df_portfolio["sp500"].iloc[0]) - 1) * 100

print("누적 수익률 (AAPL):", result_aapl)
print("누적 수익률 (S&P 500):", result_sp500)
print("-----------------------")

# 표준편차 계산
std_aapl = df_portfolio["AAPL"].pct_change().std() * np.sqrt(252)
std_sp500 = df_portfolio["sp500"].pct_change().std() * np.sqrt(252)

print("표준편차 (AAPL):", std_aapl)
print("표준편차 (S&P 500):", std_sp500)
print("-----------------------")

# 샤프 비율 계산
sharpe_aapl = df_portfolio["AAPL"].pct_change().mean() / df_portfolio["AAPL"].pct_change().std() * np.sqrt(252)
sharpe_sp500 = df_portfolio["sp500"].pct_change().mean() / df_portfolio["sp500"].pct_change().std() * np.sqrt(252)

print("샤프 비율 (AAPL):", sharpe_aapl)
print("샤프 비율 (S&P 500):", sharpe_sp500)
