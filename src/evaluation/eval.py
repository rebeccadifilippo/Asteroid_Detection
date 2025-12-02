import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os

def run_evaluation(y_test, y_pred, model_name, cmap="Blues", filename_suffix=""):
    """
    Evaluates a model, prints the classification report, and saves the confusion matrix.
    """
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # Print results
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False)
    plt.title(f'{model_name} - Confusion Matrix-{filename_suffix}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save with dynamic filename
    os.makedirs("plots", exist_ok=True)
    # e.g., plots/confusion_matrix_Main_Model_physical.png
    save_path = f"plots/confusion_matrix_{model_name}{filename_suffix}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

    return results

def compare_models(results_baseline, results_main, filename_suffix=""):
    """
    Creates a bar chart comparing the two models.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    baseline_scores = [results_baseline['accuracy'], results_baseline['precision'], results_baseline['recall'], results_baseline['f1']]
    main_scores = [results_main['accuracy'], results_main['precision'], results_main['recall'], results_main['f1']]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], baseline_scores, width, label='Baseline', color='skyblue')
    plt.bar([i + width/2 for i in x], main_scores, width, label='Main Model', color='orange')

    plt.xlabel('Metric')
    plt.ylabel('Score')
    # Add the 'f' at the start
    plt.title(f'Model Comparison {filename_suffix}')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend()
    
    # Save with dynamic filename
    save_path = f"plots/model_comparison{filename_suffix}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison plot to {save_path}")