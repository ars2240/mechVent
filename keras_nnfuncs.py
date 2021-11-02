import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import numpy as np


# def build_dataset(X: torch.Tensor, y: torch.Tensor, batch_size: int):
#     tensordata = TensorDataset(X, y)
#     dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
#     return dataloader


def get_class_weights(y_train: torch.Tensor):
    '''
    Calculates weights for imbalanced class problem. Method taken from 
    the 'balanced' option of sklearn Logistic Regression. Weights the samples
    inversely proportional to class frequency

    input: 
        y train tensor with C classes
    output: 
        vector of len C with corresponding class weights
    '''

    # quick notes
    # cast y to long for bincount because you cannot 'bin' floats
    # cast bincount vector to float to ensure that weights are calculated
    # as floats.
    wts = len(y_train) / (len(torch.unique(y_train)) *
                          torch.bincount(y_train.long()).float())

    return wts


def get_preds_labels(model, dataloader, device=torch.device('cuda')):
    '''
    Iterates over a torch dataloader object and returns model output, prediction probabily for class 1
    and a list of labels

    torch model, torch dataloader, optional device -> tensor, tensor, tensor

    Inputs:
        torch model, optimizer, train or test data, device if wanted

    Outputs:
        model output, predictions probabilities for class 1, labels

    '''

    # output of model(data)
    # appends batches
    output_lst = []
    # labels for each batch
    label_lst = []

    for batch_n, (X, y) in enumerate(dataloader):
        X = X.float().to(device)
        y = y.float().to(device)
        model.init_hidden(X)
        output = model(X)
        output = output.data
        output_lst.append(output)
        label_lst.append(y)

    # output is n x num_classes
    output_df = torch.cat(output_lst)
    y_pred = output_df[:, 1]
    y_pred = torch.sigmoid(y_pred)
    label_df = torch.cat(label_lst)

    return output_df, y_pred, label_df


def model_accuracy(y_pred: torch.Tensor, label: torch.Tensor, thresh: float):
    '''
    Given probabilities and true labels, calculates accuracy based on threshold

    tensor, tensor, float -> float

    Input:
        prediction probabilities for class 1, labels, threshold

    Output:
        accuracy between 0 and 100
    '''
    pred_class = (y_pred > thresh).float()
    acc = torch.mean((pred_class == label).float()).item() * 100
    return acc


def plot_roc(fpr, tpr, roc_auc, fpath):
    '''
    plot and save an roc curve using output of sklearn.metrics roc_curve

    Inputs:
        false pos rate vector, true pos rate vector, AUROC, filepath

    Outputs:
        None
    '''
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mechanical Ventilation ROC')
    plt.legend(loc="lower right")
    plt.savefig(fpath)
    plt.close()


def plot_loss(model_id, train_loss, val_loss):
    '''
    Plot the loss values by epoch of the train set and validation set
    and save the result


    Inputs:
        List of train loss vals, list of validation loss vals


    Output:
        none
    '''

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss,
             color='navy', label='Train Loss')
    plt.plot(range(len(val_loss)), val_loss,
             color='darkorange', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    fpath = 'results/' + model_id + '/' + model_id + '_loss.png'
    plt.savefig(fpath)
    plt.close()


def train_best_model(model, model_id, full_train_data: DataLoader, opt_epoch: int,
                     optimizer, loss, device=torch.device('cuda')):
    '''
    input:
        full train dataloader obejct, train+val sets

    output:
        model trained for optimal number of epochs found in training
        based on AUROC of validation set
    '''
    for epoch in range(opt_epoch):
        for batch_n, (X, y) in enumerate(full_train_data):
            X = X.float().to(device)
            # cross entropy loss takes an int as the target which corresponds
            # to the index of the target class
            # output should be of shape (batch_size, num_classes)
            y = y.long().to(device)

            # zero out the optimizer gradient
            # no dependency between samples
            optimizer.zero_grad()
            model.init_hidden(X)
            y_pred = model(X)
            # print(y_pred.size())
            # print(y.size())
            batch_loss = loss(y_pred, y.view(-1))
            batch_loss.backward()
            optimizer.step()

    # saving model
    model_path = 'results/{0}/{0}.pt'.format(model_id)
    # saving only the state dict
    torch.save(model.state_dict(), model_path)


def plot_test_precision_recall(fpath, label: torch.Tensor, y_pred: torch.Tensor):
    '''
    Input:
        filepath, labels, prediciton probabilities

    Output:
        None

    saves a precision recall plot to the specified filepath generated with given
    label and the predicion probabilities
    '''

    precision, recall, _ = sm.precision_recall_curve(label.cpu(), y_pred.cpu())
    plt.figure()
    plt.plot(recall, precision, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Precision Recall Curve')
    plt.savefig(fpath)
    plt.close()


def precision_recall_f1(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float):
    '''
    outputs precision, recall and f1 given a decision threshold

    Tensor, Tensor, float -> float, float, float

    Inputs:
        true values or label, prediction probabilities, decision threshold [0, 1]

    Outputs:
        precision, recall, f1-score at given threshold
    '''
    pred_class = (y_pred > threshold).float()
    y_true, pred_class = y_true.cpu(), pred_class.cpu()
    precision = sm.precision_score(y_true, pred_class)
    recall = sm.recall_score(y_true, pred_class)
    f1 = sm.f1_score(y_true, pred_class)

    return precision, recall, f1
