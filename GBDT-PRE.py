import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor  # 修改1：替换模型导入
from bayes_opt import BayesianOptimization
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# 数据加载与预处理（保持不变）
def load_and_preprocess(data_path, target_col='kobs'):
    data = pd.read_excel(data_path)
    X = data.drop(columns=[target_col])
    y = data[[target_col]]
    return X, y


# 修改2：贝叶斯优化目标函数适配GBDT
def gbdt_bayes_optimization(X_train, y_train):
    def gbdt_cv(max_depth, learning_rate, n_estimators):
        model = GradientBoostingRegressor(
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            random_state=13
        )
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
        return cv_score

    optimizer = BayesianOptimization(
        f=gbdt_cv,
        pbounds={
            'max_depth': (3, 15),
            'learning_rate': (0.01, 0.5),
            'n_estimators': (50, 200)
        },
        random_state=13
    )
    optimizer.maximize(init_points=5, n_iter=20)
    return optimizer.max['params']


# 主流程（部分修改）
if __name__ == "__main__":
    # 加载训练数据（保持不变）
    X, y = load_and_preprocess("D:/133/ML/LOAD/Al2O3/database-Al2O3.xlsx")

    # 数据分割（保持不变）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    # 标准化处理（保持不变）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 执行贝叶斯优化
    best_params = gbdt_bayes_optimization(X_train_scaled, y_train)
    print("Optimized Parameters:", best_params)

    # 修改3：使用GBDT模型训练
    model = GradientBoostingRegressor(
        max_depth=int(best_params['max_depth']),
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        random_state=13
    )
    model.fit(X_train_scaled, y_train.values.ravel())

    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    plt.bar([x for x in range(len(importance))], importance)
    plt.savefig('D:/133/ML/LOAD/Al2O3/111/importanceGBDT5.tif', dpi=300)
    plt.show()

    # 模型评估（保持不变）
    y_pred = model.predict(X_test_scaled)
    pre_train = model.predict(X_train_scaled)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

    # 预测未知数据（保持不变）
   # uk = pd.read_excel("D:/133/ML/LOAD/unknown.xlsx")
   # uk_scaled = scaler.transform(uk)
   # pre = model.predict(uk_scaled)

    # 结果保存（修改保存路径）
   # output_path = 'D:/133/ML/LOAD/UK/GBDT_pred.xlsx'  # 修改输出路径
  #  pd.DataFrame(pre, columns=['Predicted_kobs']).to_excel(output_path, index=False)
   # print(f"预测结果已保存至：{output_path}")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, color='blue', label='test')
plt.scatter(y_train, pre_train, color='red', label='train')
plt.legend()
plt.savefig('D:/133/ML/LOAD/Al2O3/111/GBDT.tif', dpi=300)
y_test = pd.DataFrame(data=y_test)
pre_test = pd.DataFrame(data=y_pred)
pre_train = pd.DataFrame(data=pre_train)
y_test.to_excel('D:/133/ML/LOAD/Al2O3/111/GBDT_test.xlsx')
pre_test.to_excel('D:/133/ML/LOAD/Al2O3/111/GBDT_pred.xlsx')
y_train.to_excel('D:/133/ML/LOAD/Al2O3/111/GBDT_train.xlsx')
pre_train.to_excel('D:/133/ML/LOAD/Al2O3/111/GBDT_pretrain.xlsx')