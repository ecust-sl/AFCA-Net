import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from matplotlib import rcParams

# --- 1. 全局绘图样式设置 (投稿标准) ---
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = 26
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 22
rcParams['legend.fontsize'] = 24


def extract_y_true(file_path):
    """从txt提取标签 (假设标签在第4列)"""
    y_true = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                y_true.append(int(parts[3]))
    return np.array(y_true)


def calculate_net_benefit(y_true, y_prob, thresholds):
    """计算DCA净收益"""
    net_benefits = []
    n = len(y_true)
    for thresh in thresholds:
        tp = np.logical_and(y_prob >= thresh, y_true == 1).sum()
        fp = np.logical_and(y_prob >= thresh, y_true == 0).sum()
        if thresh >= 1.0:
            nb = 0
        else:
            nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(nb)
    return net_benefits


# --- 2. 路径与配置 (请替换 XXX) ---
LABEL_PATH = "XXX.txt"
CSV_BASELINE = "XXX_baseline.csv"
CSV_MULTIMODAL = "XXX_multimodal.csv"
SAVE_CAL_PATH = "XXX_Calibration.tiff"
SAVE_DCA_PATH = "XXX_DCA.tiff"


def run_evaluation():
    # 数据加载
    # y_true = extract_y_true(LABEL_FILE).flatten().astype(int)
    # y_baseline = pd.read_csv(CSV_BASELINE)['binary_probability'].values
    # y_multi = pd.read_csv(CSV_MULTIMODAL)['binary_probability'].values

    # 模拟数据占位
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_baseline = np.random.rand(8)
    y_multi = np.random.rand(8)

    # --- Part A: Calibration Curve ---
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")

    # Baseline (无Label，半透明作为背景)
    fop_b, mpv_b = calibration_curve(y_true, y_baseline, n_bins=10)
    plt.plot(mpv_b, fop_b, "o--", color='black', lw=2, alpha=0.4)

    # Multimodal (突出显示)
    fop_m, mpv_m = calibration_curve(y_true, y_multi, n_bins=10)
    plt.plot(mpv_m, fop_m, "s-", color='red', label='AFCA-Multimodal', lw=3)

    plt.xlabel("Predicted Probability", fontweight='bold', labelpad=15)
    plt.ylabel("Actual Probability", fontweight='bold', labelpad=15)
    plt.legend(loc="upper left", frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.title("Calibration Curve", fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(SAVE_CAL_PATH, dpi=300, bbox_inches='tight')
    plt.show()

    # --- Part B: DCA Curve ---
    plt.figure(figsize=(10, 8))
    thresholds = np.linspace(0, 0.9, 100)

    # 基准线: All / None
    nb_all = calculate_net_benefit(y_true, np.ones(len(y_true)), thresholds)
    plt.plot(thresholds, nb_all, color='gray', linestyle='--', lw=2, label='Treat All')
    plt.axhline(y=0, color='black', linestyle='-', lw=1.5, label='Treat None')

    # Baseline (无Label)
    nb_base = calculate_net_benefit(y_true, y_baseline, thresholds)
    plt.plot(thresholds, nb_base, color='black', lw=2, alpha=0.4)

    # Multimodal
    nb_multi = calculate_net_benefit(y_true, y_multi, thresholds)
    plt.plot(thresholds, nb_multi, color='red', lw=3, label='AFCA-Multimodal')

    plt.xlim(0, 0.9)
    plt.ylim(-0.05, max(nb_multi) + 0.1)
    plt.xlabel("Threshold Probability", fontweight='bold', labelpad=15)
    plt.ylabel("Net Benefit", fontweight='bold', labelpad=15)
    plt.legend(loc="upper right", frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.title("Decision Curve Analysis", fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(SAVE_DCA_PATH, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_evaluation()