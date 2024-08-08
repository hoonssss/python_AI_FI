import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

url = "/Users/yunjaehun/desktop/유데미/fi_ai/stock_16/stock.csv"

stock_df = pd.read_csv(url)
stock_df = stock_df.sort_values(by=["Date"]);
print(stock_df);

#정규화
def normalize(df):
    x = df.copy();
    for i in x.columns[1:]:
        x[i] = x[i] / x[i][0];
    return x;

#plotly
def interactive_plot(df, title):
    fig = px.line(title = title);
    for i in df.columns[1:]:
        fig.add_scatter(x = df["Date"], y = df[i], name = i );
    fig.show();

interactive_plot(normalize(stock_df), "test");
