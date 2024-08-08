import numpy as np
import pandas as pd

url = "/Users/yunjaehun/desktop/유데미/fi_ai/stock_16/stock.csv"

stock_df = pd.read_csv(url)
stock_df = stock_df.sort_values(by=["Date"]);

np.random.seed(101);
weights = np.array(np.random.random(9));
print(weights.sum());

#정규화
weights = weights / np.sum(weights);
print(weights.sum());

#정규화
def normalize(df):
    x = df.copy();
    for i in x.columns[1:]:
        x[i] = x[i] / x[i][0];
    return x;

df_protfolio = normalize(stock_df);
print(df_protfolio);

print(df_protfolio.columns[1:]);

#stock = col, counter = index
for counter, stock in enumerate(df_protfolio.columns[1:]):
    df_protfolio[stock] = df_protfolio[stock] * weights[counter];
    df_protfolio[stock] = df_protfolio[stock] * 1_000_000;
    #df_protfolio[stock] = df_protfolio[stock] * weights[counter] * 1_000_000;

print(df_protfolio);

#Date 제외 후 sum col 추가(row 계산)
df_protfolio["test_sum"] = df_protfolio.drop(columns=["Date"]).sum(axis=1)
print(df_protfolio);

df_protfolio["%"] = 0.0000;
# 체인할당 발생
# for i in range(1,len(df_protfolio)):
#     df_protfolio["%"][i] = ((df_protfolio["test_sum"][i] - df_protfolio["test_sum"][i-1]) / df_protfolio["test_sum"][i-1]) * 100; 

for i in range(1, len(df_protfolio)):
    df_protfolio.loc[i,"%"] = ((df_protfolio["test_sum"][i] - df_protfolio["test_sum"][i-1]) / df_protfolio["test_sum"][i-1]) * 100; 
print(df_protfolio)