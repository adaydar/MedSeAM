from sklearn.metrics import auc,roc_curve

def calculate_metrics(predictions, labels):
    #print(predictions)
    #print(labels)
    TP = ((predictions == 1) & (labels == 1)).sum().item()
    TN = ((predictions == 0) & (labels == 0)).sum().item()
    FP = ((predictions == 1) & (labels == 0)).sum().item()
    FN = ((predictions == 0) & (labels == 1)).sum().item()

    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    return sensitivity, specificity, recall

def calculate_auc(predictions, labels):
    fpr, tpr, _ = roc_curve(labels, predictions)
    auc_score = auc(fpr, tpr)
    return auc_score
