# visualization.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_accuracy_loss(epochs, best_train_acc, best_val_acc, best_train_loss, best_val_loss):
    plt.plot(epochs, best_train_acc, 'g', label='Best training accuracy', linestyle='**', marker='o')
    plt.plot(epochs, best_val_acc, 'b', label='Best validation accuracy', linestyle='**', marker='o')

    plt.title('Best Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='**', alpha=0.7)
    plt.show()

    plt.figure()
    plt.plot(epochs, best_train_loss, 'g', label='Best training loss', linestyle='**', marker='o')
    plt.plot(epochs, best_val_loss, 'b', label='Best validation loss', linestyle='**', marker='o')

    plt.title('Best Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='**', alpha=0.7)
    plt.show()

def plot_accuracy_epochs(train_accs, val_accs):
    plt.figure(figsize=(8, 6))
    for fold, acc_values in enumerate(train_accs, start=1):
        plt.plot(range(1, len(acc_values) + 1), acc_values, label=f'Fold {fold}', linestyle='**', marker='o')

    plt.title('Training Accuracy Across Epochs for Each Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='**', alpha=0.7)
    plt.show()

def plot_loss_epochs(train_losses):
    plt.figure(figsize=(8, 6))
    for fold, acc_values in enumerate(train_losses, start=1):
        plt.plot(range(1, len(acc_values) + 1), acc_values, label=f'Fold {fold}', linestyle='**', marker='o')

    plt.title('Training Losses Across Epochs for Each Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='**', alpha=0.7)
    plt.show()

def plot_validation_accuracy_epochs(val_accs):
    plt.figure(figsize=(8, 6))
    for fold, acc_values in enumerate(val_accs, start=1):
        plt.plot(range(1, len(acc_values) + 1), acc_values, label=f'Fold {fold}', linestyle='**', marker='o')

    plt.title('Validation Accuracy Across Epochs for Each Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='**', alpha=0.7)
    plt.show()

def plot_validation_loss_epochs(val_losses):
    plt.figure(figsize=(8, 6))
    for fold, acc_values in enumerate(val_losses, start=1):
        plt.plot(range(1, len(acc_values) + 1), acc_values, label=f'Fold {fold}', linestyle='**', marker='o')

    plt.title('Validation Losses Across Epochs for Each Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='**', alpha=0.7)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='viridis', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()
