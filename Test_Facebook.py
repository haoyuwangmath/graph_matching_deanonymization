'''
Test Facebook networks
'''


import sys
import os
import matplotlib.pyplot as plt
import copy
import os.path as osp
import numpy as np
import scipy.io
from scipy import sparse as sp
import random
import torch
import torch_geometric.transforms as T
import argparse
import time

from GM_GNN import GM_GNN
from Generate_Graph import facebook_graph
from Statistical_Algo import D_Hop




def generate_y(num_nodes, truth):
    one_to_n = torch.arange(num_nodes)
    return [one_to_n, truth]


def test(test_set):
    model.eval()

    total_correct = 0
    total_node = 0
    for data in test_set:
        G1, G2, seeds, truth = data[0], data[1], data[2], data[3]
        num_nodes = G1.shape[0]

        Y_last, _ = model(G1, G2, seeds)
        y = generate_y(num_nodes, truth)
        correct_match = model.accuracy(Y_last, y)
        total_correct += correct_match
        total_node += num_nodes
    return total_correct / total_node


def run(num_graphs, s, L, Theta):
    facebook_filepath = "./data/facebook100"
    filedirs = os.listdir(facebook_filepath)
    result_gnn = torch.zeros(len(Theta))
    result_one_hop = torch.zeros(len(Theta))
    result_two_hop = torch.zeros(len(Theta))

    for realpath in filedirs[50: num_graphs+50]:
        # mat = scipy.io.loadmat(facebook_filepath+'/'+realpath)
        # print(realpath, mat.keys())
        for theta_i, theta in enumerate(Theta):
            dataset = []
            G1, G2, seeds, truth = facebook_graph(facebook_filepath+'/'+realpath, s, theta)
            dataset = [(G1, G2, seeds, truth)]

            n1 = G1.shape[0]
            n2 = G2.shape[0]
            eye1 = torch.eye(n1)
            eye2 = torch.eye(n2)

            # GM-GNN
            result_gnn[theta_i] += test(dataset)
            
            # 1-Hop
            result = seeds
            for _ in range(L):
                result = D_Hop(G1, G2, result)
            result_one_hop[theta_i] += sum((result[1] == truth).float()) / n1

            # 2-Hop
            G1_2 = ((((torch.mm(G1, G1))>0).float() - G1 - eye1)>0).float()
            G2_2 = ((((torch.mm(G2, G2))>0).float() - G2 - eye2)>0).float()
            result = seeds
            for _ in range(L//2):
                result = D_Hop(G1_2, G2_2, result)
            result_two_hop[theta_i] += sum((result[1] == truth).float()) / n1

    
    theta = [round(t, 4) for t in Theta.tolist()]
    result_gnn = [round(r, 4) for r in (result_gnn / num_graphs).tolist()]
    result_one_hop = [round(r, 4) for r in (result_one_hop / num_graphs).tolist()]
    result_two_hop = [round(r, 4) for r in (result_two_hop / num_graphs).tolist()]
    
    torch.set_printoptions(precision=4)
    print('Accuracy')
    print('theta ='.ljust(10), theta)
    print('GNN ='.ljust(10), result_gnn)
    print('1-Hop ='.ljust(10), result_one_hop)
    print('2-Hop ='.ljust(10), result_two_hop)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.01)

    args, unknown = parser.parse_known_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GM_GNN(num_layers=args.num_layers, hidden=args.hid).to(device)

    path = "./model/GM_GNN-pretrained.pth"
    model.load_state_dict(torch.load(path))


    start = time.time()

    L = 6
    s = 0.8
    Theta = torch.linspace(0, 0.1, steps=11)
    run(40, s, L, Theta)
                    
    end = time.time()
    print('run: ', end-start, ' s')