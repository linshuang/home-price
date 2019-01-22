"""
https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


#%%  查看一下数据
df_train = pd.read_csv('./data/train.csv')
collumns = df_train.columns

df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice']);

print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

