import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def show_results(y_test, predictions, model=None):
    miss = 0
    for predict, result in zip(predictions, y_test):
        if predict != result:
            miss += 1

    hit_rate = 1 - (miss / len(predictions))

    f1_value = f1_score(y_test, predictions)

    print("f1 is: ", f1_value)
    print("hit hate is: ", hit_rate)

    labels = ['BENIGN', 'ATTACK']
    report = classification_report(y_test, predictions, target_names=labels, digits=5)
    print("Relatório de classificação:")
    print(report)
    mat_conf = confusion_matrix(y_test, predictions)

    title = f"Confusion Matrix {model if model else ''}:"
    print(title)
    print(mat_conf)
    ConfusionMatrixDisplay(mat_conf, display_labels=labels).plot(values_format='d', cmap='Blues')

    plt.title(title)
    plt.show()
    return report