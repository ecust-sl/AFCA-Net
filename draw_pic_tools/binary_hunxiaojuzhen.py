import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams

# --- 1. 全局绘图样式设置 (论文发表标准) ---
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 22
rcParams['legend.fontsize'] = 24


def extract_y_true(file_path):
    """从txt文件中提取真实标签 (假设标签在第4列)"""
    y_true = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                y_true.append(int(parts[3]))  # 根据索引调整
    return y_true


def plot_confusion_matrix_publication(y_true, y_pred, labels=[0, 1], save_path='XXX.tiff'):
    """绘制并保存高质量混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 8))

    # 绘制热力图
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        # 内部数字加粗，字号30
        annot_kws={"size": 30, "weight": "bold", "family": "Times New Roman"},
        linewidths=1.5,
        linecolor='white',
        cbar_kws={'shrink': 0.8}
    )

    # 轴标签与刻度
    ax.set_xlabel('Predicted Labels', fontweight='bold', fontsize=26, labelpad=15)
    ax.set_ylabel('True Labels', fontweight='bold', fontsize=26, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=22)

    # 颜色条刻度
    try:
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=20)
    except IndexError:
        pass

    plt.tight_layout()
    # 导出 TIFF 格式，带 LZW 压缩
    plt.savefig(
        save_path,
        dpi=300,
        format='tiff',
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches='tight'
    )
    plt.show()


# --- 2. 数据加载部分 ---

# 路径占位符
LABEL_FILE = "XXX.txt"
PRED_CSV = "XXX.csv"
SAVE_PATH = "XXX.tiff"

# 执行流程
if __name__ == "__main__":
    # 加载真实标签
    # y_true = extract_y_true(LABEL_FILE)

    # 加载模型预测结果 (假设从 CSV 中读取结果列)
    # df = pd.read_csv(PRED_CSV)
    # y_pred = df['binary_prediction'].values

    # 示例数据占位 (正式运行时请注释掉下面两行，解除上面的注释)
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]

    # 绘图
    plot_confusion_matrix_publication(
        y_true,
        y_pred,
        labels=['Normal', 'Abnormal'],
        save_path=SAVE_PATH
    )