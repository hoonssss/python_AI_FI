import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# 유방암 데이터셋 로드
cancer = load_breast_cancer();

# 데이터셋을 DataFrame으로 변환
df_cancer = pd.DataFrame(np.c_[cancer["data"], cancer["target"]], columns=np.append(cancer["feature_names"], ["target"]));

class_1_df = df_cancer[df_cancer["target"] == 1 ] #target 1만 추출
class_0_df = df_cancer[df_cancer["target"] == 0 ] #target 0만 추출

print(class_1_df)

plt.figure(figsize=(10,7));
sns.histplot(class_1_df["mean radius"], bins=25, color="blue", kde=True);
sns.histplot(class_0_df["mean radius"], bins=25, color="red", kde=True);

plt.show();