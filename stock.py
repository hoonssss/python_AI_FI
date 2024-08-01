import pandas as pd
import numpy as np

stock_url = "/Users/yunjaehun/desktop/유데미/fi_ai/stocks.csv";
stock_result = pd.read_csv(stock_url);
print(stock_result);