import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os

def run_evaluation(y_test, y_pred, model_name, cmap="Blues", save_plots=True, filename_suffix=""):
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
    
    # Clean title formatting
    clean_suffix = filename_suffix.replace('_', ' ').strip()
    title_text = f'{model_name} - Confusion Matrix ({clean_suffix})' if clean_suffix else f'{model_name} - Confusion Matrix'
    plt.title(title_text)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save with dynamic filename
    if save_plots:
        os.makedirs("plots", exist_ok=True)
        save_path = f"plots/confusion_matrix_{model_name}{filename_suffix}.png"
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()

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
    
    clean_suffix = filename_suffix.replace('_', ' ').strip()
    title_text = f'Model Comparison ({clean_suffix})' if clean_suffix else 'Model Comparison'
    plt.title(title_text)
    
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend()
    
    save_p