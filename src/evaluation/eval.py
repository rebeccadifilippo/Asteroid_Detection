import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Ensure the folder exists
os.makedirs("plots", exist_ok=True)

def run_evaluation(y_test, y_pred, model_name="Model", cmap="Blues", save_plots=True):
    """
    Evaluate a single model using classification metrics and a confusion matrix.
    Optionally saves the confusion matrix plot to disk.
    """
    print(f"\n=== {model_name} Evaluation ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Summary metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=cmap, colorbar=False)
    plt.title(f"{model_name} - Confusion Matrix")

    if save_plots:
        filename = f"plots/confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Confusion matrix saved to {filename}")

    plt.show()

    # Return metrics for comparison
    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def compare_models(results_baseline, results_main, save_plot=True):
    """
    Compare performance between baseline and main model using a bar chart and printed summary.
    """
    print("\n=== Model Comparison ===")
    print(f"{'Metric':<12} {'Baseline':<12} {'Main Model':<12}")
    print("-" * 40)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        print(f"{metric.capitalize():<12} "
              f"{results_baseline[metric]:<12.4f} "
              f"{results_main[metric]:<12.4f}")

    # Plot side-by-side comparison
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    baseline_scores = [results_baseline[m.lower()] for m in metrics]
    main_scores = [results_main[m.lower()] for m in metrics]

    x = range(len(metrics))
    plt.figure(figsize=(7, 4))
    plt.bar(x, baseline_scores, width=0.4, label="Baseline", align='center')
    plt.bar([p + 0.4 for p in x], main_scores, width=0.4, label="Main Model", align='center')
    plt.xticks([p + 0.2 for p in x], metrics)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Model Comparison")
    plt.tight_layout()

    if save_plot:
        filename = "plots/model_comparison.png"
        plt.savefig(filename)
        print(f"Comparison chart saved to {filename}")

    plt.show()
