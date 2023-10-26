"""
pytorch 版用来跑早退点切分模型的节点端模型
"""
import argparse
import io
from queue import Queue
import queue
import select
import socket
from threading import Thread
import time

import torch
import torch.nn as nn
from node_state import NodeState, socket_recv, socket_send

import zfpy
import lz4.frame

device = 'cpu'
node_idx = -1
hostIP = '192.168.31.132'

# 树莓派实际跑的时候
# data socket port = 5000
# model & weight socket port = 5001
class TestNode:
    # def __init__(self, model_socket_port: int, data_socket_port: int) -> None:
    #     self.model_socket_port = model_socket_port
    #     self.data_socket_port = data_socket_port

    def _model_socket(self, node_state: NodeState):
        model_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # model_server.bind(('0.0.0.0', self.model_socket_port))
        model_server.bind(('0.0.0.0', 5001))
        print("[DEBUG] Model socket running, port 5001")
        model_server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10240000)
        model_server.listen(1) 
        model_cli = model_server.accept()[0]
        print("[DEBUG] Model socket accepted")
        model_cli.setblocking(0)
        model_cli.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10240000)

        # 收模型，主干模型和早退模型
        sub_model_bytes = socket_recv(model_cli, node_state.chunk_size)
        print("[DEBUG] Model socket received architecture & weights")

        # 收端口号，下个节点和早退端点
        next_node_ip = socket_recv(model_cli, chunk_size=1)
        dispatcher_early_exit_port = socket_recv(model_cli, chunk_size=1)

        # 加载模型
        sub_model = torch.jit.load(io.BytesIO(sub_model_bytes))
        sub_model.eval()
        node_state.model = sub_model

        # TODO:

        node_state.next_node = next_node_ip.decode()
        print("[DEBUG] model socket: next_node is ", node_state.next_node)
        node_state.dispatcher_port = int(dispatcher_early_exit_port.decode())
        print("[DEBUG] model socket: dispatcher EARLY EXIT port is ", node_state.dispatcher_port)
        select.select([], [model_cli], [])
        model_cli.send(b'\x06')
        model_server.close()
        print("[DEBUG] Model socket closed")
        print("[DEBUG] _model_socket Thread Finished")
    
    def _comp(self, arr):
        return lz4.frame.compress(zfpy.compress_numpy(arr))
    def _decomp(self, byts):
        return zfpy.decompress_numpy(lz4.frame.decompress(byts))

    # 接收前一个发来的
    def _data_server_socket(self, node_state: NodeState, to_send: Queue):
        data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # data_server.bind(('0.0.0.0', self.data_socket_port))
        data_server.bind(('0.0.0.0', 5000))
        data_server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10240000)
        print("[DEBUG] Data server socket running")
        data_server.listen(1) 
        data_cli = data_server.accept()[0]
        data_cli.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10240000)
        print("[DEBUG] Data server socket accepted")
        data_cli.setblocking(0)

        while True:
            x = bytes(socket_recv(data_cli, node_state.chunk_size))
            adj = bytes(socket_recv(data_cli, node_state.chunk_size))
            prev_result = bytes(socket_recv(data_cli, node_state.chunk_size))
            tup = (x, adj, prev_result)
            to_send.put(tup)
        
        print("[DEBUG] Data server socket closed")
        print("[DEBUG] Data server socket Thread Finished")

    # 发给下一个
    def _data_client_socket(self, node_state: NodeState, to_send: Queue):
        while node_state.next_node == '' or node_state.dispatcher_port == '':
            time.sleep(10)
        
        print("[DEBUG] Data client socket received next_node: ", node_state.next_node)
        sub_model = node_state.model

        next_node_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        next_node_client.connect((node_state.next_node, 5000))
        # next_node_client.connect(('localhost', node_state.next_node))
        print("[DEBUG] Data client socket connected, port", node_state.next_node)
        next_node_client.setblocking(0)

        dispatcher_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # dispatcher_client.connect(('localhost', node_state.dispatcher_port))
        print('[DEBUG] hostIP: {}, dispatcher_port: {}'.format(hostIP, node_state.dispatcher_port))
        dispatcher_client.connect((hostIP, node_state.dispatcher_port))
        print("[DEBUG] Early exit data client socket connected, port", node_state.dispatcher_port)
        dispatcher_client.setblocking(0)

        while True:
            # 在这里做节点端的推理
            with torch.no_grad():
                # 规范 to_send 的数据格式，adjs 需要是 np.array
                # x, adjs, prev_result
                x_bytes, adj_bytes, prev_result_bytes = to_send.get()
                x = torch.from_numpy(self._decomp(x_bytes)).float().to(device)
                adj = self._decomp(adj_bytes)
                prev_result = torch.from_numpy(self._decomp(prev_result_bytes)).float().to(device)

                temp_adj = [torch.from_numpy(adj[i]).to(device) for i in range((node_idx-1)*20, node_idx*20)]
                # print('len(adj):', len(temp_adj))
                # for i in range(10):
                #     print('temp_adj[{}].shape: {}'.format(i, temp_adj[i].shape))
                result, exit_num = sub_model(x, temp_adj, prev_result=prev_result)
                # 判断是否能早退
                if exit_num != -1:
                    socket_send(self._comp(result.detach().numpy()), dispatcher_client, node_state.chunk_size)
                    print("[DEBUG] EARLY EXIT at {}, result sent back to dispatcher!!!".format(exit_num))
                    continue

                print("[DEBUG] data client socket inference finished")
                # 不能早退，发给下一个节点
                print('result.shape:', result.shape, '; exit_num:', exit_num)
                # 同样规范数据格式
                socket_send(x_bytes, next_node_client, node_state.chunk_size)
                socket_send(adj_bytes, next_node_client, node_state.chunk_size)
                socket_send(self._comp(result.detach().numpy()), next_node_client, node_state.chunk_size)
                print("[DEBUG] data client socket result sent to next node")

        print("[DEBUG] Data client socket closed")
        print("[DEBUG] Data client socket Thread Finished")

    def run(self):
        node_state = NodeState(chunk_size=512*1024)
        to_send = queue.Queue(1000)
        model_thread = Thread(target=self._model_socket, args=(node_state,))
        data_server_thread = Thread(target=self._data_server_socket, args=(node_state, to_send))
        data_client_thread = Thread(target=self._data_client_socket, args=(node_state, to_send))

        model_thread.start()
        data_server_thread.start()
        data_client_thread.start()
        model_thread.join()
        data_server_thread.join()
        data_client_thread.join()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example script with argparse')
    # parser.add_argument('--model_port', type=int, default=4001, help='Port for the model (default: 4001)')
    # parser.add_argument('--data_port', type=int, default=4011, help='Port for the data (default: 4011)')
    parser.add_argument('--node_idx', type=int, default=1, help='Node index: 1, 2, 3, 4')
    args = parser.parse_args()

    # model_port = args.model_port
    # data_port = args.data_port
    node_idx = args.node_idx

    node = TestNode()
    node.run()
