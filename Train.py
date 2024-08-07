'''
Training the GNN model
'''


import os
import os.path as osp
import random

import argparse
import torch
import torch_geometric.transforms as T

from GM_GNN import GM_GNN
from Generate_Graph import synthetic_graph, facebook_graph

torch.set_printoptions(precision=4)

parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=200)

args, unknown = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GM_GNN(num_layers=args.num_layers, hidden=args.hid).to(device)


def generate_y(num_nodes, truth):
    one_to_n = torch.arange(num_nodes)
    return [one_to_n, truth]


def train(train_set, optimizer):
    model.train()

    total_loss = 0
    num_examples = 0

    for i in range(0, len(train_set), args.batch_size):
        batch = train_set[i: i+args.batch_size]
        optimizer.zero_grad()
        batch_loss = 0

        for data in batch:
            optimizer.zero_grad()

            G1, G2, seeds, truth = data[0], data[1], data[2], data[3]
            num_nodes = G1.shape[0]

            Y_last, Y_total = model(G1, G2, seeds)

            y = generate_y(num_nodes, truth)
            loss = model.loss(Y_total, y)
            batch_loss += loss
            total_loss += loss
            num_examples += 1
        
        batch_loss.backward()
        optimizer.step()
    return total_loss.item() / num_examples


@torch.no_grad()
def test(test_set):
    model.eval()

    total_correct = 0
    num_test = 0
    for data in test_set:
        G1, G2, seeds, truth = data[0], data[1], data[2], data[3]
        num_nodes = G1.shape[0]

        Y_last, _ = model(G1, G2, seeds)
        y = generate_y(num_nodes, truth)
        correct_match = model.accuracy(Y_last, y)
        total_correct += correct_match / num_nodes
        num_test += 1
    return total_correct / num_test


def run(dataset):
    model.reset_parameter()
    random.shuffle(dataset)
    train_set = dataset[:200]
    test_set = dataset[200:]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    for epoch in range(1, 1 + args.epochs):
        loss = train(train_set, optimizer)
        scheduler.step()

        if epoch % 5 == 0:
            train_accuracy = test(train_set)
            test_accuracy = test(test_set)
            print(f"epoch {epoch:03d}: Loss: {loss:.8f}, Training Accuracy: {train_accuracy:.4f}, Testing Accuracy: {test_accuracy:.4f}")
    
    final_accuracy = 100 * test(test_set)
    print("Accuracy: ", final_accuracy)
    path = "./model/GM_GNN-pretrained.pth"
    torch.save(model.state_dict(), path)

    return final_accuracy



if __name__ == '__main__':
    print("Preparing Training Data...")

    dataset = []
    graph_parameters = [
        (200, 0.1, 0.6, 0.1), (200, 0.1, 0.8, 0.1), (200, 0.1, 1.0, 0.1), (200, 0.3, 0.6, 0.1), (200, 0.3, 0.8, 0.1),
        (200, 0.3, 1.0, 0.1), (200, 0.5, 0.6, 0.1), (200, 0.5, 0.8, 0.1), (200, 0.5, 1.0, 0.1), (200, 0.02, 0.8, 0.1)
    ]
    num_graphs = 20
    for n, p, s, theta in graph_parameters:
        for _ in range(num_graphs):
            graph_pair = synthetic_graph(n, p, s, theta)
            dataset.append(graph_pair)

    s = 0.8
    theta = 0
    facebook_filepath = "./data/facebook100"
    filedirs = os.listdir(facebook_filepath)
    for realpath in filedirs[:50]:
        dataset.append(facebook_graph(facebook_filepath+'/'+realpath, s, theta, True))
    
    print("Preparation Done!")
    run(dataset)