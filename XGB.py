import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sklearn
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler  # 标准化工具
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score , KFold , cross_val_predict
from sklearn.model_selection import train_test_split
import sklearn.tree as st
import sklearn.metrics as sm
import xgboost as xgb
import sklearn.tree as tree
import sklearn.ensemble as se
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import sklearn.metrics as metrics
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
data = pd.read_excel("D:/133/ML/LOAD/AC/database-AC.xlsx")
data.head()
x = data.drop(['kobs'], axis=1)
y = data[['kobs']]
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_scaler, y, test_size=0.2, random_state=13)
def XGB_cv(max_depth, learning_rate, n_estimators):
    val = cross_val_score(
        xgb.XGBRegressor(n_estimators=int(n_estimators),
                               learning_rate=int(learning_rate),
                               max_depth=int(max_depth),
                               random_state=42,
                               oob_score = 'True'),
        x_train, y_train, scoring='neg_mean_squared_error', cv=10
    ).mean()
    return val

rf_bo = BayesianOptimization(XGB_cv,
                             {'n_estimators': (10, 500),
                              'max_depth': (2, 10),
                              'learning_rate': (0.01, 0.5)}
                             )
rf_bo.maximize(n_iter=50)
