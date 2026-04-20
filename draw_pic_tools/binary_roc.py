import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import rcParams

# --- 1. 全局样式设置 (论文发表标准) ---
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 14


def extract_y_true(file_path):
    """从文本文件中提取真实标签"""
    y_true = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                # 根据实际列索引调整 parts[3]
                y_true.append(int(parts[3]))
    return y_true


def compute_roc_auc_with_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.95):
    """计算AUC及其95%置信区间 (Bootstrap法)"""
    auc_value = roc_auc_score(y_true, y_pred)
    bootstrapped_aucs = []

    rng = np.random.RandomState(42)  # 固定随机种子
    for _ in range(n_bootstrap):
        indices = rng.choice(range(len(y_true)), size=len(y_true), replace=True)
        if len(np.unique([y_true[i] for i in indices])) < 2:
            continue
        y_true_resampled = [y_true[i] for i in indices]
        y_pred_resampled = [y_pred[i] for i in indices]
        bootstrapped_aucs.append(roc_auc_score(y_true_resampled, y_pred_resampled))

    lower = np.percentile(bootstrapped_aucs, (1 - alpha) / 2 * 100)
    upper = np.percentile(bootstrapped_aucs, (1 + alpha) / 2 * 100)
    return auc_value, lower, upper


# --- 2. 数据加载部分 (模板化) ---
# 请将 'XXX' 替换为你的实际路径
LABEL_FILE = "XXX.txt"
CSV_FILES = [
    'XXX_model1.csv',
    'XXX_model2.csv',
    'XXX_model3.csv',
    'XXX_model4.csv'
]
SAVE_PATH = 'XXX_ROC_Curve.tiff'


def run_roc_analysis():
    # 获取真实标签
    # y_true = extract_y_true(LABEL_FILE)

    # 模拟数据用于演示结构 (实际使用时请取消上方注释)
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])

    # 获取各模型预测概率 (假设列名为 'binary_probability')
    model_preds = []
    for f in CSV_FILES:
        # df = pd.read_csv(f)
        # model_preds.append(df['binary_probability'].values)
        model_preds.append(np.random.rand(len(y_true)))  # 模拟数据

    # 模型配置映射
    configs = [
        {'name': 'Baseline', 'color': 'black'},
        {'name': 'AFCA-Text', 'color': 'red'},
        {'name': 'AFCA-Image', 'color': 'orange'},
        {'name': 'AFCA-Multimodel', 'color': 'blue'}
    ]

    plt.figure(figsize=(8, 6))

    for i, pred in enumerate(model_preds):
        auc_val, low, high = compute_roc_auc_with_ci(y_true, pred)
        fpr, tpr, _ = roc_curve(y_true, pred)

        plt.plot(fpr, tpr, color=configs[i]['color'], lw=2,
                 label=f"{configs[i]['name']} (AUC = {auc_val:.3f} [{low:.3f}-{high:.3f}])")

    # 绘图修饰
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    # 高清保存
    plt.savefig(SAVE_PATH, dpi=300, format='tiff', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_roc_analysis()