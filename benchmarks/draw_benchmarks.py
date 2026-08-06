import numpy as np
import matplotlib.pyplot as plt

def draw_benchmark_results(X_test, y_true, y_pred, title_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    
    if X_test.shape[1] == 1:
        sort_idx = np.argsort(X_test[:, 0])
        ax.plot(X_test[sort_idx, 0], y_true[sort_idx], label="True Function", color="#2b5c8f", lw=2.5)
        ax.plot(X_test[sort_idx, 0], y_pred[sort_idx], label="EML Prediction", color="#e74c3c", linestyle="--", lw=2)
        ax.set_xlabel("X")
        ax.set_ylabel("y")
    else:
        ax.scatter(y_true, y_pred, alpha=0.6, color="#3498db", edgecolors="k", linewidths=0.5)
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", label="Ideal Fit (y = x)", lw=2)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")

    ax.set_title(f"Fit Quality: {title_name}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.show()