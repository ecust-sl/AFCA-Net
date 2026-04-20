import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import make_interp_spline
from matplotlib import rcParams

# --- 1. 全局样式设置 (论文发表标准) ---
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = 26
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 22
rcParams['legend.fontsize'] = 24

def extract_y_true(file_path):
    """从txt文件提取标签"""
    y_true = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                y_true.append(int(parts[3]))
    return np.array(y_true)

# --- 2. 核心绘图算法 ---

def plot_isotonic_calibration(ax, y_true, y_prob, label, color, linewidth=2, alpha=1.0):
    """使用保序回归 (Isotonic Regression) 绘制阶梯状校准曲线"""
    # 训练模型
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(y_prob, y_true)

    # 生成预测区间
    x_range = np.linspace(0, 1, 1000)
    y_isotonic = ir.predict(x_range)

    # 绘制阶梯曲线
    ax.plot(x_range, y_isotonic, color=color, label=f"{label} (Isotonic)", lw=linewidth, alpha=alpha)

def plot_smooth_calibration(ax, y_true, y_prob, label, color, linestyle='-', linewidth=2, alpha=1.0):
    """使用三次样条插值 (Spline) 绘制平滑校准曲线"""
    fop, mpv = calibration_curve(y_true, y_prob, n_bins=10)

    # 插值平滑
    mpv_new = np.linspace(mpv.min(), mpv.max(), 300)
    spl = make_interp_spline(mpv, fop, k=3)
    fop_smooth = np.clip(spl(mpv_new), 0, 1)

    ax.plot(mpv_new, fop_smooth, linestyle=linestyle, color=color, label=label, lw=linewidth, alpha=alpha)
    # 可选：绘制原始观测点
    ax.scatter(mpv, fop, color=color, s=60, alpha=0.4)

# --- 3. 路径配置 (请替换 XXX) ---
LABEL_PATH = "XXX.txt"
CSV_PATH_1 = "XXX_model1.csv"
CSV_PATH_2 = "XXX_model2.csv"
SAVE_PATH = "XXX_Calibration_Curve.tiff"

def run_main():
    # 数据获取
    # y_true = extract_y_true(LABEL_PATH)
    # y_prob_1 = pd.read_csv(CSV_PATH_1)['binary_probability'].values
    # y_prob_2 = pd.read_csv(CSV_PATH_2)['binary_probability'].values

    # 模拟数据
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 10)
    y_prob_1 = np.random.rand(80)
    y_prob_2 = np.random.rand(80)

    # 开始绘图
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 1. 理想参考线
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated", lw=2)

    # 2. 方案 A: 使用保序回归 (Isotonic) 展示阶梯效果
    plot_isotonic_calibration(ax, y_true, y_prob_2, 'AFCA-Multimodal', '#1E90FF', linewidth=4)

    # 3. 方案 B: 使用平滑插值 (Spline)
    # plot_smooth_calibration(ax, y_true, y_prob_1, 'Baseline', 'black', linestyle='--', linewidth=1.5, alpha=0.6)

    # 布局美化
    plt.xlabel("Predicted Probability", fontweight='bold', labelpad=15)
    plt.ylabel("Actual Probability", fontweight='bold', labelpad=15)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="upper left", frameon=False)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.title("Calibration Curve", fontweight='bold', pad=25)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_main()