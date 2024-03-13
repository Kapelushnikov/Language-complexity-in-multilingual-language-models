from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Union


def f1_score_func(preds: List[float], labels: List[float]) -> float:
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average="weighted")


def accuracy_per_class(
    preds: List[float], labels: List[float], label_dict: dict[Union[str, int] : int]
) -> None:
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f"Class: {label_dict_inverse[label]}")
        print(f"Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n")


def plot_confusion_matrix(
    preds: List[float], labels: List[float], label_dict: dict[Union[str, int] : int]
) -> None:
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    # Вычисляем confusion matrix
    cm = confusion_matrix(labels_flat, preds_flat)
    label_names = list(label_dict.keys())

    # Визуализация confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm[::-1],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names[::-1],
    )
    plt.ylabel("Истинные классы")
    plt.xlabel("Предсказанные классы")
    plt.show()


def metrics_per_class(
    preds: List[float], labels: List[float], label_dict: dict[Union[str, int] : int]
) -> None:
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    # Calculate confusion matrix
    cm = confusion_matrix(labels_flat, preds_flat)

    for label in np.unique(labels_flat):
        # True Positives
        TP = cm[label, label]
        # False Positives: sum of the corresponding column minus TP
        FP = np.sum(cm[:, label]) - TP
        # False Negatives: sum of the corresponding row minus TP
        FN = np.sum(cm[label, :]) - TP

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        print(f"Class: {label_dict_inverse[label]}")
        print(f"Accuracy: {TP}/{TP+FN} (True Positives / Total Actual Positives)")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}\n")
