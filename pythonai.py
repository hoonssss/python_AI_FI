import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# 유방암 데이터셋 로드
cancer = load_breast_cancer();

# 데이터셋을 DataFrame으로 변환
df_cancer = pd.DataFrame(np.c_[cancer["data"], cancer["target"]], columns=np.append(cancer["feature_names"], ["target"]));

plt.figure(figsize=(20,10));
sns.heatmap(df_cancer.corr()) #.corr 상관관계
plt.show();