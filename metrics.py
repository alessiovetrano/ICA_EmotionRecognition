from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

labels = ["Neutral", "Sad", "Fear", "Happy"]

def parse_data(filename):
    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        row = line.strip("\n").split(", ")
        data.append(np.array(row, dtype=float))

    data = np.array(data)
    return data[:, 0:4], data[:, 4]

# Confusion Matrix
def create_confusion_matrix(filename):
    pred, y = parse_data(filename)
    confusion_matrix = metrics.confusion_matrix(y, pred.argmax(1),normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    cm_display.plot()
    plt.title("Confusion Matrix".format(filename[:-12]), color="red")
    # Creazione della cartella di output se non esiste già
    output_dir = 'output/matrice_di_confusione_DGCNN_NO_ica'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_train_dgcnn_noica5.png'))


# Precision
def precision(filename):
    pred, y = parse_data(filename)
    precision = metrics.precision_score(y, pred.argmax(1), average='weighted')
    return np.round(precision, 3)  # Arrotonda alla terza cifra decimale

# Recall
def recall(filename):
    pred, y = parse_data(filename)
    recall = metrics.recall_score(y, pred.argmax(1), average='weighted')
    return np.round(recall, 3)  # Arrotonda alla terza cifra decimale

# ROC - AUC Score
def roc_auc_score(filename):
    pred, y = parse_data(filename)
    for row in pred:
        min = np.min(row)
        np.subtract(row, min, row)
        sum = np.sum(row)
        np.divide(row, sum, row)
    return np.round(metrics.roc_auc_score(y, pred, multi_class="ovr"),3)

# F1 SCORE
def f1_score(filename):
    pred, y = parse_data(filename)
    return np.round(metrics.f1_score(y, pred.argmax(1), average="weighted"),3)

# Accuracy
def accuracy(filename):
    pred, y = parse_data(filename=filename)
    return np.round(metrics.accuracy_score(y, pred.argmax(1)), 2)


# Parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the data")
    parser.add_argument("metric", type=str, choices=["cm", "roc", "f1", "acc", "precision","recall","all"], help="Metric to calculate")
    return parser.parse_args()

# Driver code
if __name__ == '__main__':
    args = parse_args()
    if args.metric == "cm":
        create_confusion_matrix(args.file)
    elif args.metric == "roc":
        print("ROC-AUC Score: {0}".format(roc_auc_score(args.file)))
    elif args.metric == "f1":
        print("F1 Score: {0}".format(f1_score(args.file)))
    elif args.metric == "acc":
        print("Accuracy: {0}".format(accuracy(args.file)))
    elif args.metric == "precision":
        print("Precision: {0}".format(precision(args.file)))
    elif args.metric == "recall":
        print("Recall: {0}".format(recall(args.file)))
    elif args.metric == "all":
        print("ROC-AUC Score: {0}".format(roc_auc_score(args.file)))
        print("F1 Score: {0}".format(f1_score(args.file)))
        print("Accuracy: {0}".format(accuracy(args.file)))
        print("Precision: {0}".format(precision(args.file)))
        print("Recall: {0}".format(recall(args.file)))



