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
from operator import *
import os
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_HOME'] = './'  # setting the environment variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m_state_dict = torch.load("./model/liuqiujun_20140621.mat", map_location=device)
xdim = [128, 62, 5]  # batch_size * channel_num * freq_num
k_adj = 40
num_out = 64
GraphModel = DGCNN(xdim, k_adj, num_out).to(device)
GraphModel.load_state_dict(m_state_dict)
for name, parameter in GraphModel.named_parameters():
    parameter.requires_grad = False
FC_Model = []
for i in range(40):
    FC_Model.append(FC_DGCNN(xdim, num_out))
    path = "./model/FC_Layer_{}".format(i + 1)
    fc_state_dict = torch.load(path, map_location=device)
    FC_Model[i].load_state_dict(fc_state_dict)
    for name, parameter in FC_Model[i].named_parameters():
        parameter.requires_grad = False


def load_DE_SEED(load_path):
    filePath = load_path
    datasets = scio.loadmat(filePath)
    DE = datasets['DE']
    dataAll = np.transpose(DE, [1, 0, 2])
    labelAll = datasets['labelAll'].flatten()

    labelAll = labelAll + 1

    return dataAll, labelAll


def load_dataloader(datatest, labeltest):
    batch_size = 64

    testiter = DataLoader(dataset=eegDataset(datatest, labeltest),
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1)

    return testiter


def train(test_iter, exitlayernum):
    for inputs, labels in test_iter:
        GraphModel.eval()
        inputs = inputs.float().to(device)
        labels = labels.to(device)
        x = inputs
        x = GraphModel.BN1(x.transpose(1, 2)).transpose(1, 2)
        l = normalize_A(GraphModel.A)
        adj = generate_cheby_adj(l, GraphModel.layer1.K, device)
        output = []
        finalresult = None
        exitnum = 0
        correct, total = 0, 0
        for i in range(len(GraphModel.layer1.gc1)):
            FC_Model[i].eval()
            if i == 0:
                result = GraphModel.layer1.gc1[i](x, adj[i])
                out = result
                out = F.relu(out)
                out = FC_Model[i](x, out)
                pred = out.argmax(dim=1)
                output.append(pred)
            else:
                result += GraphModel.layer1.gc1[i](x, adj[i])
                out = result
                out = F.relu(out)
                out = FC_Model[i](x, out)
                pred = out.argmax(dim=1)
                finalresult = pred
                same = True
                for temp in output:
                    if temp.equal(pred):
                        same = True
                    else:
                        same = False
                        break
                if same and len(output) < exitlayernum:
                    output.append(pred)
                elif (same and len(output) == exitlayernum) or i == 39:  # reach the early exit point
                    exitnum = i + 1
                    break
                else:
                    output.clear()
                    output.append(pred)
        correct += (finalresult == labels).sum().item()
        total += len(labels)
        print("The DGCNN has exit at layer {} with an acc={}".format(exitnum, correct / total))


if __name__ == '__main__':
    dir = '../SEED/DE/session1/'
    file_list = os.listdir(dir)
    sub_i = 3
    load_path = dir + file_list[sub_i]
    print("Test File: ", file_list[sub_i])
    data_test, label_test = load_DE_SEED(load_path) 
    data_test = zscore(data_test)
    criterion = nn.CrossEntropyLoss().to(device)
    test_iter = load_dataloader(data_test, label_test)
    train(test_iter, 3)
