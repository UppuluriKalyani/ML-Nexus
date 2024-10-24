import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_training_history(histories, model_name):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    for i, history in enumerate(histories):
        ax1.plot(history.history['loss'], label=f'Fold {i+1}')
        ax2.plot(history.history['accuracy'], label=f'Fold {i+1}')

    ax1.set_title(f'{model_name} - Training Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.set_title(f'{model_name} - Training Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'results/figures/{model_name.lower()}_training_history.png')
    plt.close()

def plot_roc_curve(y_test, y_pred_proba, model_name):
    """Plot ROC curve for each class"""
    plt.figure(figsize=(8, 6))
    
    for i in range(y_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'results/figures/{model_name.lower()}_roc_curve.png')
    plt.close()

def plot_pr_curve(y_test, y_pred_proba, model_name):
    """Plot Precision-Recall curve for each class"""
    plt.figure(figsize=(8, 6))
    
    for i in range(y_test.shape[1]):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, label=f'Class {i}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'results/figures/{model_name.lower()}_pr_curve.png')
    plt.close()

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'results/figures/{model_name.lower()}_confusion_matrix.png')
    plt.close()