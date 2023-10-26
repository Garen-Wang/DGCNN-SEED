"""
pytorch 版分发端代码
"""
import os
import queue
import select
import socket
from threading import Thread
from typing import List
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import zscore
from earlyexit import load_DE_SEED
from RUNearly import load_dataloader
from utils import generate_cheby_adj, normalize_A
from model import DGCNN

from node_state import socket_recv, socket_send

import lz4.frame
import zfpy

device = 'cpu'

m_state_dict = torch.load("./model/liuqiujun_20140621.mat", map_location=device)
xdim = [128, 62, 5]  # batch_size * channel_num * freq_num
k_adj = 40
num_out = 64
main_model = DGCNN(xdim, k_adj, num_out).to(device)
main_model.load_state_dict(m_state_dict)
for _, parameter in main_model.named_parameters():
    parameter.requires_grad = False
main_model.eval()

class TestDispatcher:
    def __init__(self, nodes) -> None:
        self.nodes = nodes
        self.chunk_size = 512 * 1024
        # early_exit_socket_port[-1] 替代了 data_socket_port
        self.early_exit_socket_port = [4021, 4022, 4023, 4024]
        pass

    # 用来压缩和解压 data
    def _comp(self, arr):
        return lz4.frame.compress(zfpy.compress_numpy(arr))
    def _decomp(self, byts):
        return zfpy.decompress_numpy(lz4.frame.decompress(byts))
    
    def _send_sub_models(self, sub_models: List[nn.Module]):
        for i in range(len(sub_models)):
            model_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            model_client.setblocking(0)
            model_client.settimeout(100)

            model_client.connect(('localhost', self.nodes[i][0]))
            next_node_data_port = self.nodes[i+1][1] if i != len(sub_models) - 1 else self.early_exit_socket_port[-1]

            # 发模型，这里发了主干模型和早退模型
            # model_bytes = sub_models[i].save_to_buffer()
            sub_model_bytes = sub_models[i].save_to_buffer()
            socket_send(sub_model_bytes, model_client, chunk_size=self.chunk_size)

            # 发端口，这里发了下个节点和早退回去的端口
            socket_send(str(next_node_data_port).encode(), model_client, chunk_size=1)
            socket_send(str(self.early_exit_socket_port[i]).encode(), model_client, chunk_size=1)
            select.select([model_client], [], []) # Waiting for acknowledgement: 0x06
            model_client.recv(1)

    def _data_client(self, input: queue.Queue):
        data_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # data_client.connect((self.computeNodes[0], 5000))
        data_client.connect(('localhost', self.nodes[0][1]))
        print("[DEBUG] data client connected, port ", self.nodes[0][1])
        data_client.setblocking(0)

        while True:
            x = input.get()
            # 做一些前面层，放在 dispatcher 干
            x = main_model.BN1(x.transpose(1, 2)).transpose(1, 2)
            l = normalize_A(main_model.A)
            adj = generate_cheby_adj(l, main_model.layer1.K, device)

            # 规范 to_send 的数据格式，adjs 需要是 np.array
            # x, adjs, prev_result
            zeros = np.zeros((x.size(0), 62, 64))
            adj = np.array([x.numpy() for x in adj])
            x = x.numpy()
            socket_send(self._comp(x), data_client, self.chunk_size)
            socket_send(self._comp(adj), data_client, self.chunk_size)
            socket_send(self._comp(zeros), data_client, self.chunk_size)

            print("[DEBUG] data client sent to nodes[0]")

    def _data_server(self, output: queue.Queue):
        inputs = []
        connected = []
        for port in self.early_exit_socket_port:
            data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            data_server.bind(('0.0.0.0', port))
            data_server.listen(1)
            data_server.setblocking(0)
            inputs.append(data_server)
        
        while True:
            readable, _, _ = select.select(inputs, [], [], 5)
            for sock in readable:
                if sock not in connected:
                    data_cli = sock.accept()[0]
                    print("[DEBUG] early exit result server accepted, port", port)
                    inputs.append(data_cli)
                    connected.append(data_cli)
                else:
                    data = bytes(socket_recv(sock, self.chunk_size))
                    print("result server received data")
                    pred = self._decomp(data)
                    print('pred:', pred)
                    output.put(pred)
    
    def run(self):
        sub_models = []
        SUB_MODEL_PATH = './checkpoints'
        for i in range(4):
            sub_models.append(torch.jit.load(os.path.join(SUB_MODEL_PATH, f'sub_model_{i}.pth')).to(device))
        
        self._send_sub_models(sub_models)

        input_queue = queue.Queue(10)
        output_queue = queue.Queue(10)
        
        data_client_thread = Thread(target=self._data_client, args=(input_queue,))
        data_server_thread = Thread(target=self._data_server, args=(output_queue,)) 

        data_client_thread.start()
        data_server_thread.start()

        # load dataset
        dir = '../SEED/DE/session1/'
        file_list = os.listdir(dir)
        sub_i = 3
        load_path = dir + file_list[sub_i]
        print("Test File: ", file_list[sub_i])
        data_test, label_test = load_DE_SEED(load_path) 
        data_test = zscore(data_test)
        test_iter = load_dataloader(data_test, label_test)

        # TODO: 修改一下 input_shape
        for inputs, labels in test_iter:
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            x = inputs
            input_queue.put(x)

        data_client_thread.join()
        data_server_thread.join()


dispatcher = TestDispatcher([
    (4001, 4011), (4002, 4012), (4003, 4013), (4004, 4014)
])
dispatcher.run()