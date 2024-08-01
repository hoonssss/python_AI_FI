import pandas as pd
import numpy as np

stock_url = "/Users/yunjaehun/desktop/유데미/fi_ai/stocks.csv";
stock_result = pd.read_csv(stock_url);
sorting_result = stock_result.sort_values(by=["Date"]); #Sorting Date
print(sorting_result);
print(sorting_result.columns[1:]);
print(len(sorting_result.columns[1:]));

for i in sorting_result.columns[1:]: #주식 col 반환
    print(i);

print(stock_result["sp500"].mean());
print(stock_result["sp500"].std());
print(stock_result.describe()); #통계정보를 제공

