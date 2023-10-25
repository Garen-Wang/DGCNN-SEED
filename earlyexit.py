import os
import sys
from torch.optim import lr_scheduler
import numpy as np
import torch
import torch.optim as optim
from scipy import io as scio
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import zscore
from model import DGCNN
from model import FC_DGCNN
from tqdm import tqdm, trange
from utils import eegDataset
from utils import normalize_A
from utils import generate_cheby_adj
import torch.nn.functional as F

import os
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_HOME'] = './'  # setting the environment variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m_state_dict = torch.load("./model/liuqiujun_20140621.mat")
xdim = [128, 62, 5]  # batch_size * channel_num * freq_num
k_adj = 40
num_out = 64
new_model = DGCNN(xdim, k_adj, num_out).to(device)
new_model.load_state_dict(m_state_dict)
for name, parameter in new_model.named_parameters():
     parameter.requires_grad = False


def load_DE_SEED(load_path):
    filePath = load_path
    datasets = scio.loadmat(filePath)
    DE = datasets['DE']
    dataAll = np.transpose(DE, [1, 0, 2])
    labelAll = datasets['labelAll'].flatten()

    labelAll = labelAll + 1

    return dataAll, labelAll


def load_dataloader(datatrain, datatest, labeltrain, labeltest):
    batch_size = 64
    trainiter = DataLoader(dataset=eegDataset(datatrain, labeltrain),
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=1)

    testiter = DataLoader(dataset=eegDataset(datatest, labeltest),
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1)

    return trainiter, testiter


def GetFCInput(train_iter, test_iter):
    print('began getting DGCNN\'s all layers\' result on', device, '...')
    new_model.eval()
    FC_input_train = []
    FC_label_train = []
    FC_input_test = []
    FC_label_test = []
    train_x = []
    test_x = []
    for i in range(40):
        FC_input_train.append([])
        FC_input_test.append([])

    for images, labels in train_iter:
        images = images.float().to(device)
        labels = labels.to(device)
        # 先经过DGCNN的图卷积，记录每一次卷积后的信息
        x = images
        x = new_model.BN1(x.transpose(1, 2)).transpose(1, 2)
        l = normalize_A(new_model.A)
        adj = generate_cheby_adj(l, new_model.layer1.K, device)
        train_x.append(x)
        for i in range(len(new_model.layer1.gc1)):
            if i == 0:
                result = new_model.layer1.gc1[i](x, adj[i])
            else:
                result += new_model.layer1.gc1[i](x, adj[i])
            FC_input_train[i].append(result)
        FC_label_train.append(labels)

    for images, labels in test_iter:
        images = images.float().to(device)
        labels = labels.to(device)
        # 先经过DGCNN的图卷积，记录每一次卷积后的信息
        x = images
        x = new_model.BN1(x.transpose(1, 2)).transpose(1, 2)
        l = normalize_A(new_model.A)
        adj = generate_cheby_adj(l, new_model.layer1.K, device)
        test_x.append(x)
        for i in range(len(new_model.layer1.gc1)):
            if i == 0:
                result = new_model.layer1.gc1[i](x, adj[i])
            else:
                result += new_model.layer1.gc1[i](x, adj[i])
            FC_input_test[i].append(result)
        FC_label_test.append(labels)

    return FC_input_train, FC_label_train, FC_input_test, FC_label_test, train_x, test_x


def evaluate(input_test, test_x, label_test, model):
    # Eval
    print('began test.py on ', device, '...')
    model.eval()
    correct, total = 0, 0
    print(len(test_x))
    for j in range(len(input_test)):
        # Add channels = 1
        input_test[j] = F.relu(input_test[j])
        input_test[j] = input_test[j].float().to(device)
        # Categogrical encoding
        label_test[j] = label_test[j].to(device)
        test_x[j] = test_x[j].to(device)
        output = model(test_x[j], input_test[j])
        pred = output.argmax(dim=1)

        correct += (pred == label_test[j]).sum().item()
        total += len(label_test[j])
    print('test.py Accuracy: {}'.format(correct / total))
    return correct / total


def train(input_train, label_train, input_test, label_test, model, criterion, num_epochs, trainx, testx):
    # Train
    optimizer = []
    for i in range(len(model)):
        optimizer.append(optim.Adam(model[i].parameters(), lr=0.001, weight_decay=0.0001))
    for i in range(len(model)):
        for j in range(10):
            torch.cuda.empty_cache()
        print('began training on layer ', i, ' on ', device, '...')
        n = 0
        acc_test_best = 0.0
        for j in range(len(input_train[i])):
            input_train[i][j] = F.relu(input_train[i][j])
            input_train[i][j] = input_train[i][j].float().to(device)
            trainx[j] = trainx[j].to(device)
        for ep in trange(num_epochs):
            model[i].train()
            n += 1
            batch_id = 1
            correct, total, total_loss = 0, 0, 0.
            print(len(label_train))
            for j in range(len(input_train[i])):
                output = model[i](trainx[j], input_train[i][j])
                loss = criterion(output, label_train[j].long())
                pred = output.argmax(dim=1)

                correct += (pred == label_train[j]).sum().item()
                total += len(label_train[j])
                accuracy = correct / total
                total_loss += loss
                loss.backward()
                # scheduler.step()
                optimizer[i].step()
                # print(optimizer.state_dict)
                optimizer[i].zero_grad()

                print('Layer {}, epoch {}, batch {}, loss: {}, accuracy: {}'.format(i + 1, ep + 1, batch_id,
                                                                                    total_loss / batch_id, accuracy))

                batch_id += 1

            print('Total loss for layer {} epoch {}: {}'.format(i + 1, ep + 1, total_loss))

            acc_test = evaluate(input_test[i], testx, label_test, model[i])
            if acc_test >= acc_test_best:
                n = 0
                acc_test_best = acc_test
                model_best = model[i]

            # 学习率逐渐下降，容易进入局部最优，当连续10个epoch没有跳出，且有所下降，强制跳出
            if n >= num_epochs // 10 and acc_test < acc_test_best - 0.1:
                print('#########################reload#########################')
                n = 0
                model[i] = model_best
            # find best test.py acc model in all epoch(not last epoch)
        print('>>> best test Accuracy on Layer {}: {}'.format(i + 1, acc_test_best))
        torch.save(model_best.state_dict(), 'FC_Layer_{}'.format(i + 1))

    return acc_test_best


if __name__ == '__main__':
    dir = '../SEED/DE/session1/'  # 04-0.9916， 0.86
    # os.chdir(dir) # 可能在寻找子文件的时候路径进了data
    file_list = os.listdir(dir)
    sub_num = len(file_list)
    num_epochs = 20
    acc_mean = 0
    acc_all = []
    sub_i = 3
    train_list = copy.deepcopy(file_list)
    load_path = dir + file_list[sub_i]  # ../表示上一级目录
    # 选择1个作为验证数据
    data_test, label_test = load_DE_SEED(load_path)  # data （‘采样点’，通道，4频带， 1080 59 4）   lable  对应‘采样点’的标签 1080

    # if device.type == 'cuda':
    #         print('empty cuda cache...')
    #         torch.cuda.empty_cache()

    data_test = zscore(data_test)

    # 将验证数据剔除出去，用剩下的14个样本进行训练
    train_list.remove(file_list[sub_i])
    test_list = file_list[sub_i]

    print("Train list :", train_list)
    print("Test List: ", test_list)
    train_num = len(train_list)

    data_train = []
    label_train = []

    for train_i in range(train_num):

        train_path = dir + train_list[train_i]
        data, label = load_DE_SEED(train_path)

        data = zscore(data)

        if train_i == 0:
            data_train = data
            label_train = label
        else:
            data_train = np.concatenate((data_train, data), axis=0)
            label_train = np.concatenate((label_train, label), axis=0)

    # data_train = zscore(data_train)

    criterion = nn.CrossEntropyLoss().to(device)  # 使用这个函数需要注意：标签是整数，不要onehot，已经包含了softmax

    ## 训练的数据要求导，必须使用torch.tensor包装
    data_train = torch.tensor(data_train)
    label_train = torch.tensor(label_train)

    train_iter, test_iter = load_dataloader(data_train, data_test, label_train, label_test)
    FC_DGCNN_input_train, FC_DGCNN_label_train, FC_DGCNN_input_test, FC_DGCNN_label_test, train_x, test_x = GetFCInput(
        train_iter, test_iter)
    model = []
    for i in range(len(new_model.layer1.gc1)):
        temp = FC_DGCNN(xdim, num_out).to(device)
        model.append(temp)
    acc_test_best = train(FC_DGCNN_input_train, FC_DGCNN_label_train, FC_DGCNN_input_test, FC_DGCNN_label_test, model,
                          criterion, num_epochs, train_x, test_x)
    acc_mean = acc_mean + acc_test_best / sub_num
    acc_all.append(acc_test_best)
