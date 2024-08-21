import numpy as np
import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go

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

#beta, alpha
beta, alpha = np.polyfit(stock_daily_return["sp500"], stock_daily_return["AAPL"], 1);
print("beta {}, alpha {}".format(beta,alpha));
print("---------------------------------");

# plot
# stock_daily_return.plot(kind = "scatter", x = "sp500", y = "AAPL");
# plt.plot(stock_daily_return["sp500"], beta * stock_daily_return["sp500"] + alpha, '-', color = 'r');
# plt.show();

# 연 수익
rm = stock_daily_return["sp500"].mean() * 252
print(rm);
print("---------------------------------");

# 무위험 이자율
rf = 0;

# CAPL(AAPL)
ER_AAPL = rf + (beta * (rm-rf));
print(ER_AAPL);
print("---------------------------------");

# all stock beta calculate
beta = {};
alpha = {};
for i in stock_daily_return.columns:
    if i != "Date" and i != "sp500":
        plt.figure(figsize=(10, 5))
        # 베타와 알파 계산
        b, a = np.polyfit(stock_daily_return["sp500"], stock_daily_return[i], 1)
       
        # 베타와 알파 값을 딕셔너리에 저장
        beta[i] = b
        alpha[i] = a

        # 회귀선 플로팅
        plt.plot(stock_daily_return["sp500"], b * stock_daily_return["sp500"] + a, "-", color="r", label=f"Fit Line: Beta={b:.2f}, Alpha={a:.2f}")
        
        # 산점도 플로팅
        plt.scatter(stock_daily_return["sp500"], stock_daily_return[i], alpha=0.5, label=f"{i} Data")
        
        # 레이블 및 제목 설정
        plt.xlabel("SP500 Daily Return")
        plt.ylabel(i)
        plt.title(i + " : SP500")
        plt.legend()
        plt.grid(True)
    
        # 플롯 표시
        #plt.show()

# 베타와 알파 출력
print("Beta values:", beta)
print("Alpha values:", alpha)
print("---------------------------------");

beta_keys = list(beta.keys());
print(beta_keys);
print(type(beta_keys));
print("---------------------------------");
ER = {};
rf = 0;
#연수익률
rm = stock_daily_return["sp500"].mean() * 252;
for i in beta_keys:
    ER[i] = rf + (beta[i] * (rm-rf))

for i in ER:
    print("{} : {}".format(i, ER[i]));

#동일한 비중을 갖기 위하여
portfolio_weights = 1/8 * np.ones(8);
print(portfolio_weights);
print(ER.values());
ER_portfolio = sum(list(ER.values()) * portfolio_weights);
print("동일한 비중을 투자할 시 수익률 {}".format(ER_portfolio));
aapl_tsla = 0.5 * ER["AAPL"] + 0.5 * ER["AMZN"];
print("50% 50% 비율로 투자 시 AAPL + AMZN {}".format(aapl_tsla));
print("---------------------------------");