import random
from PIL import ImageFilter
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)

## freezing
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_backbone(model):
    num_train_params = count_parameters(model)
    print(f"Train params before freezing: {num_train_params}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    num_train_params = count_parameters(model)
    print(f"Train params after freezing: {num_train_params}")
    return model
        
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)
    
    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        class_accs = 0
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs


def F_measure(preds, labels, openset=False, theta=None):
    
    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.
        
        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro') * 100

def mic_acc_cal(preds, labels):
    # Micro accuracy
    acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1 * 100

def calculate_metrics(pred, target, target_names):
    res = []
    cls_report = classification_report(
        y_true=np.array(target),
        y_pred=np.array(pred),
        target_names=target_names,
    )

    return {
        'report': cls_report,
        'accuracy': accuracy_score(target, pred),
        'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
        'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
        'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
        'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
        'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
        'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
    }
