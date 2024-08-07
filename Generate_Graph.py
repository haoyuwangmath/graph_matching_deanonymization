"""
Generate various graphs for training and testing
"""

import sys

import numpy as np
import torch
import scipy.io



def synthetic_graph(n, p, s, theta, degree_sorted=False):
    # Parent graph G(n,p)
    parent = (torch.rand(n,n) < p).float()
    parent = torch.triu(parent, diagonal=1)
    parent = parent + parent.T

    # Subsampling
    subsampling = (torch.rand(n,n) < s).float()
    subsampling = torch.triu(subsampling, diagonal=1)
    subsampling = subsampling + subsampling.T
    G1 = parent * subsampling
    subsampling = (torch.rand(n,n) < s).float()
    subsampling = torch.triu(subsampling, diagonal=1)
    subsampling = subsampling + subsampling.T
    G2 = parent * subsampling

    # ground truth
    if degree_sorted:
        degree = parent.sum(0)
        _, indices = torch.sort(degree, descending=True)
        G1 = G1[indices,:][:,indices]
        G2 = G2[indices,:][:,indices]
        truth = n-1-torch.arange(n)
    else:
        truth = torch.randperm(n)
    
    G1 = G1[truth,:][:,truth]
    num_seeds = int(n * theta)
    seed_indices = torch.randperm(n)[:num_seeds]
    seeds = [seed_indices, truth[seed_indices]]

    return (G1, G2, seeds, truth)



def facebook_graph(realpath, s, theta, subsample=True):
    mat = scipy.io.loadmat(realpath)
    adj = torch.tensor(mat['A'].toarray()).float()
    N = adj.shape[0]
    n = N
    if subsample:
        n = 200
        indices = torch.randperm(N)[:n]
        adj = adj[indices,:][:,indices]
    
    subsampling = (torch.rand(n,n) < s).float()
    subsampling = torch.triu(subsampling, diagonal=1)
    subsampling = subsampling + subsampling.T
    G1 = adj * subsampling

    subsampling = (torch.rand(n,n) < s).float()
    subsampling = torch.triu(subsampling, diagonal=1)
    subsampling = subsampling + subsampling.T
    G2 = adj * subsampling

    truth = torch.randperm(n)
    G1 = G1[truth,:][:,truth]

    num_seeds = int(n * theta)
    seed_indices = torch.randperm(n)[:num_seeds]
    seeds = [seed_indices, truth[seed_indices]]

    return (G1, G2, seeds, truth)



def read_edges(file_path):
    data = np.loadtxt(file_path, dtype=int)
    edges = [(row[0], row[1]) for row in data]
    return edges


def build_adjacency_matrix(edges):
    nodes = set()
    for edge in edges:
        nodes.update(edge)
    node_list = sorted(nodes)
    node_index = {node: idx for idx, node in enumerate(node_list)}

    n = len(node_list)
    adjacency_matrix = np.zeros((n, n), dtype=int)

    for node1, node2 in edges:
        i = node_index[node1]
        j = node_index[node2]
        adjacency_matrix[i, j] = 1
        # adjacency_matrix[j, i] = 1  # For undirected graph

    return adjacency_matrix, node_list


def load_graph(realpath, s, theta, subsample=True):
    edges = read_edges(realpath)
    adj_matrix, node_list = build_adjacency_matrix(edges)
    adj = torch.tensor(adj_matrix).float()
    N = adj.shape[0]
    n = N
    if subsample:
        n = 200
        indices = torch.randperm(N)[:n]
        adj = adj[indices,:][:,indices]
    
    subsampling = (torch.rand(n,n) < s).float()
    subsampling = torch.triu(subsampling, diagonal=1)
    subsampling = subsampling + subsampling.T
    G1 = adj * subsampling

    subsampling = (torch.rand(n,n) < s).float()
    subsampling = torch.triu(subsampling, diagonal=1)
    subsampling = subsampling + subsampling.T
    G2 = adj * subsampling

    truth = torch.randperm(n)
    G1 = G1[truth,:][:,truth]

    num_seeds = int(n * theta)
    seed_indices = torch.randperm(n)[:num_seeds]
    seeds = [seed_indices, truth[seed_indices]]

    return (G1, G2, seeds, truth)