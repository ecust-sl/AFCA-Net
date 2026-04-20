import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams

# --- Global Style Settings for Publication ---
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = 26
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 22
rcParams['legend.fontsize'] = 24


def plot_confusion_matrix_three_class(y_true, y_pred, labels=[0, 1, 2], save_path='XXX.tiff'):
    """
    Optimized confusion matrix plotting function for academic publication.
    """
    # 1. Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # 2. Setup Figure
    plt.figure(figsize=(9, 8))

    # 3. Create Heatmap
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 30, "weight": "bold"},  # Internal numbers
        cbar_kws={'shrink': 0.85}
    )

    # 4. Axis Labels and Ticks
    ax.set_xlabel('Predicted Labels', fontweight='bold', fontsize=26, labelpad=15)
    ax.set_ylabel('True Labels', fontweight='bold', fontsize=26, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=22)

    # Keep labels horizontal
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # 5. Colorbar Font Size
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)

    # 6. High-Quality Save
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        format='tiff',
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches='tight'
    )
    plt.show()


# --- Data Loading Section (Template) ---

# Replace 'XXX' with your actual file paths
file_path_labels = "XXX.txt"
csv_file_predictions = "XXX.csv"


def load_data():
    """
    Placeholder for your data loading logic.
    Ensure y_true and y_pred are 1D arrays or lists of the same length.
    """
    # Example loading logic (Simplified)
    # df = pd.read_csv(csv_file_predictions)
    # y_pred = df['column_name'].values

    # Placeholder data for template execution
    y_true_placeholder = [0, 1, 2, 0, 1, 2]
    y_pred_placeholder = [0, 1, 2, 0, 2, 1]

    return y_true_placeholder, y_pred_placeholder


if __name__ == "__main__":
    # 1. Load your data
    y_true, y_pred = load_data()

    # 2. Define save path
    output_path = 'XXX.tiff'

    # 3. Plot
    plot_confusion_matrix_three_class(
        y_true,
        y_pred,
        labels=['Class 0', 'Class 1', 'Class 2'],
        save_path=output_path
    )