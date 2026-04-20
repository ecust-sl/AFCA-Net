import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score, calibration_curve
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize
from matplotlib import rcParams

# --- 1. 全局绘图样式设置 ---
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = 26
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 22
rcParams['legend.fontsize'] = 18


# --- 2. 核心计算函数 ---

def compute_auc_with_ci(y_true, y_score, n_iterations=1000, ci_level=0.95):
    """计算 AUC 及 Bootstrap 置信区间 (支持合并计算)"""
    y_true_f = y_true.flatten()
    y_score_f = y_score.flatten()

    auc_value = roc_auc_score(y_true_f, y_score_f)
    bootstrapped_scores = []

    for _ in range(n_iterations):
        indices = resample(np.arange(len(y_true_f)), replace=True)
        if len(np.unique(y_true_f[indices])) < 2: continue
        bootstrapped_scores.append(roc_auc_score(y_true_f[indices], y_score_f[indices]))

    sorted_scores = np.sort(bootstrapped_scores)
    lower = sorted_scores[int((1 - ci_level) / 2 * len(sorted_scores))]
    upper = sorted_scores[int((1 + ci_level) / 2 * len(sorted_scores))]
    return auc_value, lower, upper


def calculate_net_benefit(y_true, y_prob, thresholds):
    """DCA 净收益计算"""
    n = len(y_true)
    net_benefits = []
    for thresh in thresholds:
        tp = np.logical_and(y_prob >= thresh, y_true == 1).sum()
        fp = np.logical_and(y_prob >= thresh, y_true == 0).sum()
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh)) if thresh < 1 else 0
        net_benefits.append(nb)
    return net_benefits


# --- 3. 绘图函数模板 ---

def plot_final_roc(y_data_list, pred_data_list, names, colors, save_path='XXX.tiff'):
    """绘制对比 ROC 曲线"""
    plt.figure(figsize=(8, 6))
    for y_true, y_pred, name, color in zip(y_data_list, pred_data_list, names, colors):
        auc_val, low, high = compute_auc_with_ci(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true.flatten(), y_pred.flatten())
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC={auc_val:.3f} [{low:.3f}-{high:.3f}])')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_calibration_dca(y_true, y_prob, model_name='Model', save_prefix='XXX'):
    """绘制校准曲线与 DCA"""
    # Calibration
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    fop, mpv = calibration_curve(y_true.flatten(), y_prob.flatten(), n_bins=10)
    plt.plot(mpv, fop, "s-", color='red', label=model_name, lw=3)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Probability")
    plt.legend()
    plt.savefig(f'{save_prefix}_cal.tiff', dpi=300)
    plt.show()

    # DCA
    plt.figure(figsize=(10, 8))
    thresh = np.linspace(0, 0.9, 100)
    nb = calculate_net_benefit(y_true.flatten(), y_prob.flatten(), thresh)
    plt.plot(thresh, nb, color='red', lw=3, label=model_name)
    plt.axhline(y=0, color='gray', linestyle='-', label='None')
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.legend()
    plt.savefig(f'{save_prefix}_dca.tiff', dpi=300)
    plt.show()


# --- 4. 数据加载与主逻辑 (模板) ---

def load_and_process():
    # 路径占位符
    CSV_PATHS = ["XXX_1.csv", "XXX_2.csv", "XXX_3.csv", "XXX_4.csv"]
    LABEL_PATH = "XXX.txt"

    # 这里应包含你原有的 extract_y_true_from_txt 和 pd.read_csv 逻辑
    # 最终输出应为 list of numpy arrays
    # 示例数据占位:
    y_true_mock = label_binarize([0, 1, 2, 1], classes=[0, 1, 2])
    y_pred_mock = np.random.rand(4, 3)

    return y_true_mock, y_pred_mock


if __name__ == "__main__":
    # 1. 获取数据
    y_true_bin, y_pred = load_and_process()

    # 2. 定义展示配置
    models = ["Baseline", "AFCA-Text", "AFCA-Image", "AFCA-Multimodel"]
    colors = ["black", "red", "orange", "blue"]

    # 3. 绘图调用 (示例)
    # plot_final_roc([y_true_bin]*4, [y_pred]*4, models, colors)
    # plot_calibration_dca(y_true_bin, y_pred, model_name="AFCA-Multimodel")