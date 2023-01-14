import os, sys
import numpy as np
from utils import *
import torch
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
from torchvision.transforms import Normalize
import torch.nn as nn
import torch.nn.functional as F
from models import MyResNet, ConvNet
from tqdm import tqdm
from torch.utils.data import TensorDataset
from copy import deepcopy
import argparse
import pandas as pd

from fairlearn.reductions import GridSearch, EqualizedOdds, ExponentiatedGradient
from fairlearn.metrics import (
    MetricFrame, equalized_odds_difference, equalized_odds_ratio,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='fairness')
parser.add_argument('--method', type=str, default='M2')
parser.add_argument('--ft_epoch', type=int, default=1000)
parser.add_argument('--ft_lr', type=float, default=1e-2)
parser.add_argument('--alpha', type=float, default=2.)
parser.add_argument('--constraint', type=str, default='EO')
parser.add_argument('--seed', type=int, default=202212)
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--checkpoint', type=str, default=None)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

######################
# Data Preprocessing #
######################

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

print("Load Data ...")

#########################
# CelebA dataset (could be slow!)
data_root = args.data_path
train_dataset = datasets.CelebA(data_root, split="train", target_type=["attr"], transform=transform)
valid_dataset = datasets.CelebA(data_root, split="valid", target_type=["attr"], transform=transform)
test_dataset = datasets.CelebA(data_root, split="test", target_type=["attr"], transform=transform)

TRAIN_BS = 1024
TEST_BS = 2048

print(f'Train Size: {len(train_dataset)}, Validation Size: {len(valid_dataset)}, Test Size: {len(test_dataset)}')

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BS,
                                          shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valid_dataset, batch_size=TEST_BS,
                                        shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BS,
                                         shuffle=False, num_workers=4)


def train_per_epoch(model, optimizer, criterion, epoch, num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, (images, labels) in enumerate(trainloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # move to GPU
        images, labels = images.to(device), labels[:, 9].to(device)

        # forward
        outputs = model.forward(images)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels).item()

    epoch_loss /= len(trainloader)
    epoch_acc /= len(train_dataset)

    print('TRAINING Epoch %d/%d Loss %.4f Accuracy %.4f' % (epoch, num_epochs, epoch_loss, epoch_acc))


def valid_per_epoch(model, epoch, num_epochs, criterion):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, (images, labels) in enumerate(valloader):
        # move to GPU
        images, labels = images.to(device), labels[:, 9].to(device)

        # forward
        outputs = model.forward(images)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)

        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels).item()

    epoch_loss /= len(valloader)
    epoch_acc /= len(valid_dataset)

    print('VALID Epoch %d/%d Loss %.4f Accuracy %.4f' % (epoch, num_epochs, epoch_loss, epoch_acc))

    return epoch_loss


def Finetune(model, criterion, trainloader, valloader, testloader):
    model.eval()

    best_clf = None
    method = args.method

    ###################################################
    # Option 1 (B1): Directly test
    # Option 2 (B2): Finetune on validation dataset
    # Option 3 (B3): Finetune on balanced-sampled (training dataset + validation dataset)
    # Option 4 (M1): Finetune on validation dataset + constraint
    # Option 5 (M2): Finetune on balanced-sampled + constraint
    ###################################################

    ################
    # Prepare the dataset
    ################
    x_train, y_train, a_train = prepare_data(trainloader, model, device)
    x_test, y_test, a_test = prepare_data(testloader, model, device)
    x_finetune, y_finetune, a_finetune = prepare_data(valloader, model, device)

    if method == 'B1':
        x_finetune = x_train
        y_finetune = y_train
        a_finetune = a_train

    elif method == 'B2' or method == 'M1':
        pass

    elif method == 'B3' or method == 'M2':  # Sample a balanced dataset
        X = torch.cat([x_train, x_finetune])
        hair = torch.cat([y_train, y_finetune])
        gender = torch.cat([a_train, a_finetune])
        g_idx = []
        g_idx.append(torch.where((gender + hair) == 2)[0])  # (1, 1)
        g_idx.append(torch.where((gender + hair) == 0)[0])  # (0, 0)
        g_idx.append(torch.where((gender - hair) == 1)[0])  # (1, 0)
        g_idx.append(torch.where((gender - hair) == -1)[0])  # (0, 1)
        for i, g in enumerate(g_idx):
            idx = torch.randperm(g.shape[0])
            g_idx[i] = g[idx]
        min_g = min([len(g) for g in g_idx])
        print(min_g)
        temp_g = torch.cat([g[:min_g] for g in g_idx])
        x_finetune = X[temp_g]
        y_finetune = hair[temp_g]
        a_finetune = gender[temp_g]

    #############
    # Fine-tune #
    #############
    model.train()
    model.set_grad(False)
    model.append_last_layer()
    model = model.to(device)
    optimizer = optim.SGD(model.out_fc.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5, last_epoch=-1)
    finetune_dataset = TensorDataset(x_finetune, y_finetune, a_finetune)
    # For B3 and M2, considering balance, only tried full batch
    finetuneloader = torch.utils.data.DataLoader(finetune_dataset, batch_size=6300, shuffle=True)
    print(len(finetune_dataset))

    losses = []
    trigger_times = 0
    best_loss = 1e9
    for epoch in range(1, args.ft_epoch + 1):
        epoch_loss = 0.0
        epoch_loss_fairness = 0.0
        epoch_acc = 0.0
        for batch_idx, (x, y, a) in enumerate(finetuneloader):
            x, y, a = x.to(device), y.to(device), a.to(device)
            optimizer.zero_grad()
            outputs = model.out_fc(x)
            log_softmax, softmax = F.log_softmax(outputs, dim=1), F.softmax(outputs, dim=1)
            if method == 'M1' or method == 'M2':  # Use the fairness constraint
                if args.constraint == 'MMF':
                    loss = mmf_constraint(criterion, log_softmax, y, a)
                else:
                    if args.constraint == 'EO':
                        fpr, fnr = eo_constraint(softmax[:, 1], y, a)
                        loss_fairness = fpr + fnr
                    # elif args.constraint == 'DP':
                    #     loss_fairness = dp_constraint(softmax[:, 1], a)
                    elif args.constraint == 'DI':
                        loss_fairness = di_constraint(softmax[:, 1], a)
                    elif args.constraint == 'AE':
                        loss_fairness = ae_constraint(criterion, log_softmax, y, a)
                    epoch_loss_fairness += loss_fairness.item()
                    loss_1 = criterion(log_softmax, y)
                    loss = loss_1 + args.alpha * loss_fairness
            else:
                loss = criterion(log_softmax, y)
            
            epoch_loss += loss.item()

            loss.backward(retain_graph=True)
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)
            epoch_acc += torch.sum(preds == y).item()

        scheduler.step()

        epoch_loss /= len(finetuneloader)
        epoch_loss_fairness /= len(finetuneloader)
        epoch_acc /= len(finetune_dataset)
        losses.append(epoch_loss)
        print('FINETUNE Epoch %d/%d   Loss_1: %.4f   Loss_2: %.4f   Accuracy: %.4f' % (
            epoch, args.ft_epoch, epoch_loss, epoch_loss_fairness, epoch_acc))

        # Early Stop
        # if (epoch > 50) and (losses[-1] >= losses[-2]):
        #     trigger_times += 1
        #     if trigger_times > 2:
        #         break
        # else:
        #     trigger_times = 0

    #     if epoch_loss < best_loss and epoch > 20:
    #         best_model = deepcopy(model)
    #         best_loss = epoch_loss

    # model = best_model

    model.eval()

    ######
    # Test
    ######
    def get_pred(x):  # Avoid exceeding the memory limit
        dataset = TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=TEST_BS, shuffle=False)
        outs = []
        for x in loader:
            out = model.out_fc(x[0].to(device)).cpu().detach().numpy()
            outs.append(out)
        outs = np.concatenate(outs)
        pred = np.argmax(outs, 1)
        return pred

    pred_train = get_pred(x_train)
    pred_finetune = get_pred(x_finetune)
    pred_test = get_pred(x_test)
    sensitive_attrs = ['gender']
    y_train, y_finetune, y_test = y_train.numpy(), y_finetune.numpy(), y_test.numpy()
    a_train, a_finetune, a_test = a_train.numpy(), a_finetune.numpy(), a_test.numpy()
    a_train, a_finetune, a_test = {'gender': a_train}, {'gender': a_finetune}, {'gender': a_test}

    def train_test_classifier():
        print_clf_stats(pred_train, pred_finetune, pred_test, y_train, a_train, y_finetune, a_finetune, y_test, a_test,
                        sensitive_attrs)

        finetune_eod = equalized_odds_difference(y_finetune, pred_finetune, sensitive_features=a_finetune)
        finetune_eor = equalized_odds_ratio(y_finetune, pred_finetune, sensitive_features=a_finetune)
        print("\n")
        print(f'Finetune equalized_odds_difference: {finetune_eod}')
        print(f'Finetune equalized_odds_ratio: {finetune_eor}')

        test_eod = equalized_odds_difference(y_test, pred_test, sensitive_features=a_test)
        test_eor = equalized_odds_ratio(y_test, pred_test, sensitive_features=a_test)
        print("\n")
        print(f'Test equalized_odds_difference: {test_eod}')
        print(f'Test equalized_odds_ratio: {test_eor}')

        finetune_did = disparity_impact_difference(y_finetune, pred_finetune, sensitive_features=a_finetune)
        finetune_dir = disparity_impact_ratio(y_finetune, pred_finetune, sensitive_features=a_finetune)
        print("\n")
        print(f'Finetune disparity_impact_difference: {finetune_did}')
        print(f'Finetune disparity_impact_ratio: {finetune_dir}')

        test_did = disparity_impact_difference(y_test, pred_test, sensitive_features=a_test)
        test_dir = disparity_impact_ratio(y_test, pred_test, sensitive_features=a_test)
        print("\n")
        print(f'Test disparity_impact_difference: {test_did}')
        print(f'Test disparity_impact_ratio: {test_dir}')

        finetune_aed = accuracy_equality_difference(y_finetune, pred_finetune,
                                                    sensitive_features=a_finetune['gender'])
        finetune_aer = accuracy_equality_ratio(y_finetune, pred_finetune, sensitive_features=a_finetune['gender'])
        print("\n")
        print(f'Finetune accuracy_equality_difference: {finetune_aed}')
        print(f'Finetune accuracy_equality_ratio: {finetune_aer}')

        test_aed = accuracy_equality_difference(y_test, pred_test, sensitive_features=a_test['gender'])
        test_aer = accuracy_equality_ratio(y_test, pred_test, sensitive_features=a_test['gender'])
        print("\n")
        print(f'Test accuracy_equality_difference: {test_aed}')
        print(f'Test accuracy_equality_ratio: {test_aer}')

        finetune_mmf = max_min_fairness(y_finetune, pred_finetune, sensitive_features=a_finetune['gender'])
        print("\n")
        print(f'Finetune max_min_fairness: {finetune_mmf}')

        test_mmf = max_min_fairness(y_test, pred_test, sensitive_features=a_test['gender'])
        print("\n")
        print(f'Test max_min_fairness: {test_mmf}')

    print("== Constrainted ==")
    train_test_classifier()
    print("\n-----------------------------------------------------------------------------------\n")


def main():
    model = MyResNet(num_classes=2, pretrain=False)
    # model = ConvNet(num_classes=2)
    model = model.cuda()
    criterion = nn.NLLLoss()

    if args.checkpoint is not None:
        print('Recovering from %s ...' % (args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        #########
        # Training #
        #########
        NUM_EPOCHS = 100
        losses = []
        trigger_times = 0
        best_loss = 1e9
        best_model = None
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, last_epoch=-1)
        for epoch in range(NUM_EPOCHS):
            train_per_epoch(model, optimizer, criterion, epoch + 1, NUM_EPOCHS)
            epoch_loss = valid_per_epoch(model, epoch + 1, NUM_EPOCHS, criterion)
            losses.append(epoch_loss)
            if epoch_loss < best_loss and epoch > 5:
                best_model = deepcopy(model)
                best_loss = epoch_loss
            # Early Stop
            if (epoch > 20) and (losses[-1] >= losses[-2]):
                trigger_times += 1
                if trigger_times > 2:
                    break
            else:
                trigger_times = 0

            scheduler.step()

        checkpoint = {"model_state_dict": best_model.state_dict()}
        torch.save(checkpoint, "./res_checkpint.pkl")
        model = best_model

    ################
    # Finetune and test #
    ################
    Finetune(model, criterion, trainloader, valloader, testloader)


if __name__ == '__main__':
    main()
