import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def run_evaluation(baseline_results, main_results):
    # Baseline confusion matrix
    print("\n--- Baseline Confusion Matrix ---")
    ConfusionMatrixDisplay.from_predictions(
        baseline_results["y_test"],
        baseline_results["y_pred"],
        cmap="Blues"
    )
    plt.show()

    # Main model confusion matrix
    print("\n--- Main Model Confusion Matrix ---")
    ConfusionMatrixDisplay.from_predictions(
        main_results["y_test"],
        main_results["y_pred"],
        cmap="Greens"
    )
    plt.show()
