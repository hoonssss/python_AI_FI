import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stock_url = "/Users/yunjaehun/desktop/유데미/fi_ai/stocks.csv";
stock_result = pd.read_csv(stock_url);
sorting_result = stock_result.sort_values(by=["Date"]); #Sorting Date

def show_plot(df, title):
    df.plot(x = "Date", title = title, linewidth = 3, figsize = (15,7))
    plt.grid();
    plt.show();

def normalize(df):
    x = df.copy();
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x;

print(normalize(sorting_result));
show_plot(normalize(sorting_result),"title");
