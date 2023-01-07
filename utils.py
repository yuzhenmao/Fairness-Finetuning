import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, \
    average_precision_score
from fairlearn.metrics import (
    MetricFrame, equalized_odds_difference, equalized_odds_ratio,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)


def prepare_data(dataloader, model, device):
    x_, y_, a_ = [], [], []
    for batch_idx, (images, labels) in enumerate(dataloader):
        gender = labels[:, 20]
        a_.append(gender)
        hair = labels[:, 9]
        y_.append(hair)
        X = model.get_features(images.to(device)).detach().cpu()
        x_.append(X)

    return torch.cat(x_), torch.cat(y_), torch.cat(a_)


def eo_constraint(p, y, a):
    fpr = torch.abs(torch.sum(p * (1 - y) * a) / torch.sum(a) - torch.sum(p * (1 - y) * (1 - a)) / torch.sum(1 - a))
    fnr = torch.abs(torch.sum((1 - p) * y * a) / torch.sum(a) - torch.sum((1 - p) * y * (1 - a)) / torch.sum(1 - a))
    return fpr, fnr


def di_constraint(p, a):
    di = -1 * torch.min((torch.sum(a * p) / torch.sum(a)) / (torch.sum((1 - a) * p) / torch.sum((1 - a))),
                        (torch.sum((1 - a) * p) / torch.sum((1 - a))) / (torch.sum(a * p) / torch.sum(a)))
    return di


def dp_constraint(p, a):
    dp = torch.abs((torch.sum(a * p) / torch.sum(a)) - (torch.sum((1 - a) * p) / torch.sum((1 - a))))
    return dp


def ae_constraint(criterion, log_softmax, y, a):
    loss_p = criterion(log_softmax[a == 1], y[a == 1])
    loss_n = criterion(log_softmax[a == 0], y[a == 0])
    return torch.abs(loss_p - loss_n)


def mmf_constraint(criterion, log_softmax, y, a):
    # loss_p = criterion(log_softmax[a == 1], y[a == 1])
    # loss_n = criterion(log_softmax[a == 0], y[a == 0])
    # return torch.max(loss_p, loss_n)
    y_p_a = y + a
    y_m_a = y - a
    loss_1 = criterion(log_softmax[y_p_a == 2], y[y_p_a == 2])  # (1, 1)
    loss_2 = criterion(log_softmax[y_p_a == 0], y[y_p_a == 0])  # (0, 0)
    loss_3 = criterion(log_softmax[y_m_a == 1], y[y_m_a == 1])  # (1, 0)
    loss_4 = criterion(log_softmax[y_m_a == -1], y[y_m_a == -1])  # (0, 1)
    return torch.max(torch.max(loss_1, loss_2), torch.max(loss_3, loss_4))


def disparity_impact_difference(y, pred, sensitive_features):
    return demographic_parity_difference(y, pred, sensitive_features=sensitive_features)


def disparity_impact_ratio(y, pred, sensitive_features):
    return demographic_parity_ratio(y, pred, sensitive_features=sensitive_features)


def accuracy_equality_difference(y, pred, sensitive_features):
    misclassification_rate_p = sum(y[sensitive_features == 1] != pred[sensitive_features == 1]) / sum(
        sensitive_features == 1)
    misclassification_rate_n = sum(y[sensitive_features == 0] != pred[sensitive_features == 0]) / sum(
        sensitive_features == 0)
    return abs(misclassification_rate_p - misclassification_rate_n)


def accuracy_equality_ratio(y, pred, sensitive_features):
    misclassification_rate_p = sum(y[sensitive_features == 1] != pred[sensitive_features == 1]) / sum(
        sensitive_features == 1)
    misclassification_rate_n = sum(y[sensitive_features == 0] != pred[sensitive_features == 0]) / sum(
        sensitive_features == 0)
    return min(misclassification_rate_p / (misclassification_rate_n + 1e-6),
               misclassification_rate_n / (misclassification_rate_p + 1e-6))


def max_min_fairness(y, pred, sensitive_features):
    # classification_rate_p = sum(y[sensitive_features == 1] == pred[sensitive_features == 1]) / sum(
    #     sensitive_features == 1)
    # classification_rate_n = sum(y[sensitive_features == 0] == pred[sensitive_features == 0]) / sum(
    #     sensitive_features == 0)
    # return min(classification_rate_p, classification_rate_n)
    y_p_a = y + sensitive_features
    y_m_a = y - sensitive_features
    classification_rate_1 = sum(y[y_p_a == 2] == pred[y_p_a == 2]) / sum(y_p_a == 2)
    classification_rate_2 = sum(y[y_p_a == 0] == pred[y_p_a == 0]) / sum(y_p_a == 0)
    classification_rate_3 = sum(y[y_m_a == 1] == pred[y_m_a == 1]) / sum(y_m_a == 1)
    classification_rate_4 = sum(y[y_m_a == -1] == pred[y_m_a == -1]) / sum(y_m_a == -1)
    return min(min(classification_rate_1, classification_rate_2), min(classification_rate_3, classification_rate_4))


def print_fpr_fnr_sensitive_features(y_true, y_pred, x_control, sensitive_attrs):
    for s in sensitive_attrs:
        s_attr_vals = x_control[s]
        print("||  s  || FPR. || FNR. ||")
        for s_val in sorted(list(set(s_attr_vals))):
            y_true_local = y_true[s_attr_vals == s_val]
            y_pred_local = y_pred[s_attr_vals == s_val]

            # acc = float(sum(y_true_local==y_pred_local)) / len(y_true_local)

            # fp = sum(np.logical_and(y_true_local == 0.0, y_pred_local == +1.0)) # something which is -ve but is misclassified as +ve
            # fn = sum(np.logical_and(y_true_local == +1.0, y_pred_local == 0.0)) # something which is +ve but is misclassified as -ve
            # tp = sum(np.logical_and(y_true_local == +1.0, y_pred_local == +1.0)) # something which is +ve AND is correctly classified as +ve
            # tn = sum(np.logical_and(y_true_local == 0.0, y_pred_local == 0.0)) # something which is -ve AND is correctly classified as -ve

            # all_neg = sum(y_true_local == 0.0)
            # all_pos = sum(y_true_local == +1.0)

            # fpr = float(fp) / float(fp + tn)
            # fnr = float(fn) / float(fn + tp)
            # tpr = float(tp) / float(tp + fn)
            # tnr = float(tn) / float(tn + fp)
            fpr = false_positive_rate(y_true_local, y_pred_local)
            fnr = false_negative_rate(y_true_local, y_pred_local)

            if isinstance(s_val, float):  # print the int value of the sensitive attr val
                s_val = int(s_val)
            print("||  %s  || %0.2f || %0.2f ||" % (s_val, fpr, fnr))


def print_clf_stats(pred_train, pred_finetune, pred_test, y_train, a_train, y_finetune, a_finetune, y_test, a_test,
                    sensitive_attrs):
    train_acc, finetune_acc, test_acc = accuracy_score(y_train, pred_train), accuracy_score(y_finetune,
                                                                                            pred_finetune), accuracy_score(
        y_test, pred_test)
    train_auc, finetune_auc, test_auc = roc_auc_score(y_train, pred_train), roc_auc_score(y_finetune,
                                                                                          pred_finetune), roc_auc_score(
        y_test, pred_test)

    for s_attr in sensitive_attrs:
        print("*** Train ***")
        print("Accuracy: %0.3f, AUC: %0.3f" % (train_acc, train_auc))
        print_fpr_fnr_sensitive_features(y_train, pred_train, a_train, sensitive_attrs)

        print("\n")
        print("*** Finetune ***")
        print("Accuracy: %0.3f, AUC: %0.3f" % (finetune_acc, finetune_auc))
        print_fpr_fnr_sensitive_features(y_finetune, pred_finetune, a_finetune, sensitive_attrs)

        print("\n")
        print("*** Test ***")
        print("Accuracy: %0.3f, AUC: %0.3f" % (test_acc, test_auc))
        print_fpr_fnr_sensitive_features(y_test, pred_test, a_test, sensitive_attrs)
        print("\n")
