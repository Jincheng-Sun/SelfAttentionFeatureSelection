import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

plt.rcParams['savefig.dpi'] = 300  # pixel
plt.rcParams['figure.dpi'] = 300  # resolution
plt.rcParams["figure.figsize"] = [5, 4] # figure size

def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_test, probas_pred=y_pred)
    area = auc(recalls, precisions)
    return area, precisions, recalls, thresholds

def plot_pr_curve(recalls, precisions, auc, x_axis = 1):
    plt.rcParams['savefig.dpi'] = 300  # pixel
    plt.rcParams['figure.dpi'] = 300  # resolution
    plt.rcParams["figure.figsize"] = [5, 4]  # figure size

    plt.plot(recalls, precisions, color="darkorange", label='Precision-Recall curve (area = %0.3f)' % auc)
    plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, x_axis])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    return plt

def auc_roc(y_pred, y_test):
    auc = roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds

def plot_roc_curve(fprs, tprs, auc, x_axis = 1):

    plt.plot(fprs, tprs, color="darkorange", label='ROC curve (area = %0.3f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, x_axis])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return plt

def multiclass_metrics(y_real, y_pred):
    acc = accuracy_score(y_real, y_pred)
    cm = confusion_matrix(y_real, y_pred)
    report = classification_report(y_real, y_pred, digits=5)
    return acc, cm, report

def plot_cm(cm):
    plt.clf()
    df = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[0]))
    sn.heatmap(df, annot=True)
    return plt