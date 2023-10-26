import os
import sys
from typing import List, Optional
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
    FC_Model.append(FC_DGCNN(xdim, num_out).to(device))
    path = "./model/FC_Layer_{}".format(i + 1)
    fc_state_dict = torch.load(path, map_location=device)
    FC_Model[i].load_state_dict(fc_state_dict)
    for name, parameter in FC_Model[i].named_parameters():
        parameter.requires_grad = False


class SplitedGraphModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.is_final = False
    
    # 返回 中间结果，是否早退
    def forward(self, x, adj: List[torch.Tensor], prev_result: Optional[torch.Tensor]=None):
        # assert len(self.layers) == 10
        # assert len(self.exits) == 10
        # assert len(adj) == 10
        outputs = []
        exit_num = -1
        # FIXME: to(device=torch.device('cuda')) 不应该出现在这里吧？
        result = torch.zeros((x.size(0), 62, 64)).to(device=torch.device('cuda')) if prev_result is None else prev_result # 先写死
        print('prev_result.shape:', result.shape)

        for i, (layer, exit) in enumerate(zip(self.layers, self.exits)):
            temp = layer(x, adj[i])
            print('temp.shape:', temp.shape)
            result += temp
            # result += layer(x, adj[i])
            print('x: {}, result: {}'.format(x.dtype, result.dtype))
            out = exit(x, F.relu(result))
            pred = out.argmax(dim=1)

            if i == 0:
                outputs.append(pred)
            elif exit_num == -1:
                # 还没退出
                same = all(output.equal(pred) for output in outputs)
                if same and len(outputs) < 3:
                    # 相同的层不超过3
                    outputs.append(pred)
                elif same and len(outputs) == 3:
                    # 相同的层有3层，或者最后一个退出点
                    exit_num = i + 1
                elif self.is_final and i == 10 - 1:
                    # 整个推理的最后一个出口了
                    exit_num = i + 1
                    outputs = [pred]
                else:
                    # 不同
                    outputs = [pred]
        return outputs[-1] if exit_num != -1 else result, exit_num


# 分成4份，每份10个GCN
def split_graph_model(graph_model: nn.Module, fc_models: List[nn.Module]) -> List[nn.Module]:
    assert len(GraphModel.layer1.gc1) == 40
    sub_models = []
    for i in range(0, 40, 10):
        sub_model = SplitedGraphModel()
        for j in range(i, i + 10):
            sub_model.layers.append(graph_model.layer1.gc1[j])
            sub_model.exits.append(fc_models[j])
        sub_models.append(sub_model)
    sub_models[-1].is_final = True
    return sub_models

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


def local_inference(test_iter, exitlayernum):
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
                print(result.shape)
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

def local_test_inference(main_model, sub_models, test_iter):
    main_model.eval()
    for sub_model in sub_models:
        sub_model.eval()
    for inputs, labels in test_iter:
        inputs = inputs.float().to(device)
        labels = labels.to(device)
        x = inputs
        x = main_model.BN1(x.transpose(1, 2)).transpose(1, 2)
        l = normalize_A(main_model.A)
        adj = generate_cheby_adj(l, main_model.layer1.K, device)
        correct, total = 0, 0
        result = None
        pred = None
        exit_num = -1
        for i, sub_model in enumerate(sub_models):
            result, exit_num = sub_model(x, adj[i*10:(i+1)*10], prev_result=result)
            if exit_num != -1:
                pred = result
                exit_num = i*10 + exit_num
                break
        correct += (pred == labels).sum().item()
        total += len(labels)
        print("The DGCNN has exit at layer %d with an acc=%.3f%%" % (exit_num, 100.0 * correct / total))

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

    sub_models = split_graph_model(GraphModel, FC_Model)
    CKPT_PATH = './checkpoints'
    for i, sub_model in enumerate(sub_models):
        torch.jit.script(sub_model).save(os.path.join(CKPT_PATH, f'sub_model_{i}.pth'))
    print('saved')
    
    loaded_sub_models = []
    for i in range(len(sub_models)):
        loaded_sub_models.append(torch.jit.load(os.path.join(CKPT_PATH, f'sub_model_{i}.pth')).to(device))
    
    local_test_inference(GraphModel, loaded_sub_models, test_iter)
    # local_inference(test_iter, 3)
