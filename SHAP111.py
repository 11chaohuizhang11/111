import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# 可视化样式设置
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 中文系统建议使用宋体/SimSun
    'axes.unicode_minus': False,  # 解决负号显示问题
    'figure.dpi': 300  # 提高输出分辨率
})
shap.initjs()


def shap_analysis(data_path, target_col='kobs', save_path='D:/133/ML/LOAD/Al2O3/111/'):
    """执行完整的SHAP分析流程"""
    try:
        # 数据准备
        df = pd.read_excel(data_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 数据集分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 模型训练（可根据需要替换为其他树模型）
        model = GradientBoostingRegressor(n_estimators=220, max_features=3, max_depth=2, max_leaf_nodes=2,
                                          random_state=42)
        model.fit(X_train, y_train)

        # SHAP解释器初始化
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        def custom_colormap():
            colors = ["#a6cee3", "#FFFFFF", "#fcc5c0"]  # 浅蓝-淡紫-浅粉
            return LinearSegmentedColormap.from_list("custom", colors)

        # 可视化分析
        # 1. 全局特征重要性
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title("Global feature importance rank", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(f"{save_path}global_importance.tif", bbox_inches='tight', dpi=600, pil_kwargs={'compression': 'none'})

        # 2. 特征效应分布图（重点修改部分）
        plt.figure(figsize=(12, 8), facecolor='none')

        # 使用自定义颜色映射
        cmap = custom_colormap()

        # 绘制小提琴图
        #shap.plots.beeswarm(shap_values, color=cmap, plot_type="bar")
        shap.summary_plot(
            shap_values,
            plot_type="violin",
            cmap=custom_colormap(),
        )

        # 获取当前坐标轴
        ax = plt.gca()
        # 调整坐标轴标签大小
        ax.tick_params(axis='x', labelsize=18)  # X轴刻度
        ax.tick_params(axis='y', labelsize=20)  # Y轴特征名称（调大更清晰）
        plt.xticks(rotation=0, ha='right')  # 可选：X刻度倾斜防止重叠
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(18)
        # 定位颜色条对象（通常位于第二个axes）
        fig = plt.gcf()
        cax = fig.axes[1]

        # 设置颜色条标签格式
        cax.tick_params(axis='y',
                        labelsize=20,  # 标签大小
                        length=3,  # 刻度线长度
                        width=1)  # 刻度线宽度
        cax.set_ylabel("SHAP value",
                       fontsize=20,
                       rotation=270,
                       labelpad=20)
        plt.title("Characteristic SHAP value distribution", fontsize=24, pad=15)
        # 去除网格线和边框装饰
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')  # 保留底边但颜色调浅
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{save_path}feature_distribution.tif", bbox_inches='tight', dpi=600, transparent=True, pil_kwargs={'compression': 'none'})

        # 3. 单样本解释（交互式HTML）
        shap.save_html(f"{save_path}individual_explanation.html",
                       shap.force_plot(explainer.expected_value,
                                       shap_values.values[0, :],
                                       X_test.iloc[0, :],
                                       feature_names=X.columns.tolist()))


        print("SHAP分析完成，结果已保存至指定目录")

    except Exception as e:
        print(f"执行出错: {str(e)}")

# 使用示例（需替换实际路径）
shap_analysis(
    data_path='D:/133/ML/LOAD/Al2O3/database-Al2O3.xlsx',
    target_col='kobs',
    save_path='D:/133/ML/LOAD/Al2O3/111'
)