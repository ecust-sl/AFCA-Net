import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- 1. 全局绘图样式设置 (论文发表标准) ---
rcParams['font.family'] = 'Times New Roman'


# --- 2. 绘图核心函数 ---
def create_publication_radar(data, title, filename, color_map, labels):
    """
    创建高质量学术雷达图
    """
    num_vars = len(labels)
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合曲线

    fig, ax = plt.subplots(figsize=(10, 11), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 绘制每个模型的指标
    for model_name, values in data.items():
        vals = values + values[:1]  # 闭合数据
        ax.plot(angles, vals, color=color_map.get(model_name, '#000000'), lw=3.5,
                label=model_name, marker='o', markersize=9,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        # 背景填充设为透明，突出线条
        ax.fill(angles, vals, color='white', alpha=0.0)

    # 旋转坐标轴使第一个指标位于正上方
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 极坐标标签设置
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', which='major', pad=25)

    # 极轴刻度设置 (百分比数值)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.set_yticklabels(["60%", "80%", "100%"], fontsize=24, color="#333333", fontweight='bold')

    # 背景网格强化
    ax.grid(True, linestyle='--', alpha=0.8, color='#888888', linewidth=1.5)
    ax.spines['polar'].set_visible(False)

    # 标题
    fig.suptitle(title, x=0.5, y=0.94, size=30, fontweight='bold')

    # 底部图例设置
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=26,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.3
    )

    # 导出设置
    plt.tight_layout(rect=[0.05, 0.12, 0.95, 0.92])
    plt.savefig(filename, format='tiff', dpi=300, bbox_inches='tight')
    plt.show()


# --- 3. 配置与执行部分 (请在此替换路径和数据) ---

def run_main():
    # 指标标签
    METRIC_LABELS = ['Sensitivity', 'Specificity', 'Accuracy', 'AUC', 'F1']

    # 模型配色方案
    COLOR_MAP = {
        'Baseline': '#2E8B57',
        'AFCA-Text': '#FF4500',
        'AFCA-Image': '#FFD700',
        'AFCA-Multimodel': '#1E90FF'
    }

    # 示例数据结构 (请将 XXX 逻辑替换为你的读取逻辑)
    # data 格式应为: { 'ModelName': [val1, val2, val3, val4, val5] }
    train_metrics = {
        'Baseline': [0.71, 0.84, 0.73, 0.85, 0.71],
        'AFCA-Multimodel': [0.78, 0.89, 0.82, 0.88, 0.81]
    }

    # 路径占位符
    SAVE_TRAIN_PATH = "XXX_Radar_Train.tiff"
    SAVE_TEST_PATH = "XXX_Radar_Test.tiff"

    # 执行绘图
    create_publication_radar(train_metrics, "Train Set", SAVE_TRAIN_PATH, COLOR_MAP, METRIC_LABELS)


if __name__ == "__main__":
    run_main()