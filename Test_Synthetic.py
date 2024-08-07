"""
Test synthetic correlated Erdos-Renyi graphs
"""

import sys
import os.path as osp
import numpy as np
import torch
import time
import argparse

from GM_GNN import GM_GNN
from Generate_Graph import synthetic_graph
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


def run(n, p, s, L, Theta, Iteration):
    result_gnn = torch.zeros(Iteration, len(Theta))
    result_one_hop = torch.zeros(Iteration, len(Theta))
    result_two_hop = torch.zeros(Iteration, len(Theta))
    

    for theta_i, theta in enumerate(Theta):
        for j in range(Iteration):
            G1, G2, seeds, truth = synthetic_graph(n, p, s, theta)
            dataset = [(G1, G2, seeds, truth)]
            
            # GM_GNN
            result_gnn[j, theta_i] = test(dataset)

            # 1-Hop
            result = seeds
            for _ in range(L):
                result = D_Hop(G1, G2, result)
            result_one_hop[j, theta_i] = sum((result[1] == truth).float()) / n

            # 2-Hop
            eye_n = torch.eye(n)
            G1_2 = ((((torch.mm(G1, G1))>0).float() - G1 - eye_n)>0).float()
            G2_2 = ((((torch.mm(G2, G2))>0).float() - G2 - eye_n)>0).float()
            result = seeds
            for _ in range(L//2):
                result = D_Hop(G1_2, G2_2, result)
            result_two_hop[j, theta_i] = sum((result[1] == truth).float()) / n
    

    gnn_std, result_gnn = torch.std_mean(result_gnn, dim=0, unbiased=False)
    one_hop_std, result_one_hop = torch.std_mean(result_one_hop, dim=0, unbiased=False)
    two_hop_std, result_two_hop = torch.std_mean(result_two_hop, dim=0, unbiased=False)

    theta = [round(t, 4) for t in Theta.tolist()]
    result_gnn = [round(r, 4) for r in result_gnn.tolist()]
    result_one_hop = [round(r, 4) for r in result_one_hop.tolist()]
    result_two_hop = [round(r, 4) for r in result_two_hop.tolist()]

    torch.set_printoptions(precision=4)
    print(f'Parameters: n={n}, p={p}, s={s}, L={L}')
    print('Accuracy')
    print('theta ='.ljust(10), theta)
    print('GNN ='.ljust(10), result_gnn)
    print('1-Hop ='.ljust(10), result_one_hop)
    print('2-Hop ='.ljust(10), result_two_hop)
    

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=6)
    args, unknown = parser.parse_known_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GM_GNN(num_layers=args.num_layers, hidden=args.hid).to(device)

    path = "./model/GM_GNN-pretrained.pth"
    model.load_state_dict(torch.load(path))
    for param in model.parameters():
        param.requires_grad = False
    
    start = time.time()
    # Sparse
    L = 6
    n = 500
    p = 0.01
    s = 0.8
    Theta = torch.linspace(0, 0.2, steps=11)
    Iteration = 10
    run(n, p, s, L,Theta, Iteration)

    print('------------------------------------------------------------------')
    
    # Dense
    L = 6
    n = 500
    p = 0.2
    s = 0.8
    Theta = torch.linspace(0, 0.05, steps=11)
    Iteration = 10
    run(n, p, s, L, Theta, Iteration)

    end = time.time()
    print('run: ', end-start, ' s')