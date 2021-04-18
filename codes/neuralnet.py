import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import copy

def test_error(model, device, testloader):
    error = 0
    count = 0
    with torch.no_grad():
        for input, output in testloader:
            x_batch = input.to(device)
            y_batch = output.to(device)
            y_predict = model(x_batch)
            useless, maxed = torch.max(y_predict, dim=1)
            count += len(maxed)
            error += (maxed == y_batch).sum().item()
        return error/count

class simple_dataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        img, label = self.x_data[index], self.y_data[index]
        return img, label

    def __len__(self):
        return len(self.x_data)


class Block(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.BatchNorm1d(inchannel),
            nn.Linear(inchannel, outchannel),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential()
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        return out

class neuralnet(nn.Module):
    def __init__(self, input):
        super(neuralnet, self).__init__()
        self.neck = 8
        self.width = 1024
        self.bn1 = nn.BatchNorm1d(input)
        self.fc1 = nn.Linear(input,self.width)
        self.fcs = nn.Sequential(*[Block(self.width,self.width) for i in range(self.neck)])
        self.fc4 = nn.Linear(self.width, 7)
        self.sc = nn.Linear(input,7)

    def forward(self, ip):
        x = F.relu(self.fc1(self.bn1(ip.float())))
        x = self.fcs(x)

        out = self.fc4(x)
        return out

def train(model, traindataset, testset, epoch, batch_size, weight_decay, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    dataloader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
    train_acc = []
    test_acc = []
    for i in range(epoch):
        tot_loss = 0
        count = 0
        tt_acc = 0
        acc_count = 0
        for input, output in dataloader:
            x_batch = input.to(device)
            y_batch = output.to(device)
            y_predict = model(x_batch)
            y_predict = F.softmax(y_predict, dim=1)
            loss = nn.CrossEntropyLoss()(y_predict, y_batch.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            useless, maxed = torch.max(y_predict, dim=1)
            count += 1
            acc_count += len(maxed)
            tot_loss += loss.item()
            tt_acc += (maxed == y_batch).sum().item()
        tnl = tt_acc/acc_count
        train_acc.append(tnl)
        tstl = test_error(model, device, testloader)
        test_acc.append(tstl)
        if i %10==0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95
            print("{} / {}, , train loss is {}, tain acc is {}, test acc is {}".format(i, epoch, tot_loss/count, tnl, tstl))
    return train_acc, test_acc


def logistic_regression(traindatax, traindatay, testdatax, testdatay):
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(traindatax, traindatay)
    train_acc = logreg.score(traindatax,traindatay)
    test_acc = logreg.score(testdatax, testdatay)
    print("logistic regression: train error rate is {}, test error rate is {}".format(1-train_acc, 1-test_acc))

def linear_regression(traindatax, traindatay, testdatax, testdatay):
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import Ridge
    lr = MultiOutputRegressor(Ridge(random_state=123))
    ty = prepare_oh(traindatay,7)
    tsty = prepare_oh(testdatay,7)
    lr.fit(traindatax, ty)
    ypred = lr.predict(traindatax)
    ytestpred = lr.predict(testdatax)
    train_acc = np.mean(np.argmax(ypred,axis=1)==traindatay)
    test_acc = np.mean(np.argmax(ytestpred,axis=1)==testdatay)
    print("linear regression: train error rate is {}, test error rate is {}".format(1-train_acc, 1-test_acc))


def prepare_oh(labels,max_label):
    return labels
    n = len(labels)
    oh = np.zeros((n,max_label))
    for i in range(n):
        oh[i,labels[i]]=1
    return oh


if __name__=="__main__":
    data_raw = pd.read_excel(r'../DryBeanDataset/Dry_Bean_Dataset.xlsx')#.to_numpy()
    input = data_raw.iloc[:,:-1].to_numpy()
    class_map = {'SEKER':0, 'BARBUNYA':1, 'BOMBAY':2, 'CALI':3,
                 'HOROZ':4, 'SIRA':5, 'DERMASON':6, }
    label = np.array([class_map[name] for name in data_raw.iloc[:,-1]])
    size = len(label)
    test_idx = np.random.choice(size, size//5, replace=True)
    #print(test_idx)
    train_idx = np.delete(np.arange(size), test_idx, axis=None)
    #logistic_regression(input[train_idx], label[train_idx], input[test_idx], label[test_idx])
    #linear_regression(input[train_idx], label[train_idx], input[test_idx], label[test_idx])

    #---neural network---
    device = torch.device('cuda')
    model = neuralnet(len(input[0])).to(device)
    train_y = prepare_oh(label[train_idx],7)
    traindataset = simple_dataset(torch.tensor(input[train_idx]), torch.tensor(train_y))
    test_y = prepare_oh(label[test_idx],7)
    testdataset = simple_dataset(torch.tensor(input[test_idx]), torch.tensor(test_y))
    trainerr, testerr = train(model, traindataset, testdataset, 100, 128, 1e-3, device)
    fig, ax = plt.subplots(1, 1)
    ax.plot(list(range(len(trainerr))), trainerr, label='train accuracy')
    ax.plot(list(range(len(testerr))), testerr, label='test accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


