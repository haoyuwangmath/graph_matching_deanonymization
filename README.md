# Social Network De-anonymization via Graph Matching
This repository discusses graph matching algorithms for the social network de-anonymization problem, which refers to finding the same users between different social networks that are possibly anonymized. Compared with other previous works that leverages user profile (e.g. user id), our approaches only rely on graph topological information.

## Graph Matching Algorithms
Graph matching refers to the problem of finding the optimal vertex correspondence between two graphs such that the two graphs after the mapping shares the maximum number of edges. In mathematical language, using $A$ and $B$ to denote the adjacency matrices of graphs $G_1$ and $G_2$, we aim to solve quadratic assignment problem

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?\max_{\pi%20\in%20S_n}%20\sum_{i,j=1}^n%20A_{ij}B_{\pi(i)\pi(j)}" alt="\max_{\pi \in S_n} \sum_{i,j=1}^n A_{ij}B_{\pi(i)\pi(j)}">
</p>

We have two types of graph matching methods: statistical algorithms and GNN-based deep learning algorithms. To briefly compare these two categories of approaches, statistical algorithms only ,

### Statistical Algorithms
* Umeyama's spectral method: this spectral algorithm use correlations between ordered eigenvectors to construct a matrix of similarity score.
  
  <p align="center">
  <img src="https://latex.codecogs.com/png.latex?X%20=%20%5Csum_%7Bi%3D1%7D%5En%20%7Cu_i%7C%20%7Cv_i%5E%5Ctop%7C" alt="X = \sum_{i=1}^n |u_i| |v_i^\top|">
  </p>
  
  where $\{u_i\}$ are the eigenvectors of $A$ and $\{v_i\}$ are the eigenvectors of $B$. Then it solves a linear assignment problem of the similarity score $X$ to find the optimal correspondence
  
* **Pairwise spectral alignment**: this algorithm is similar to Umeyama's algorithm, but it considers correlations between all pairs of eigenvectors. The similarity matrix is now a weighted sum of these pairwise correlations whose weights are determined by the corresponding eigenvalue gaps.

  <p align="center">
  <img src="https://latex.codecogs.com/png.latex?X%20%3D%20%5Csum_%7Bi%2Cj%3D1%7D%5En%20%5Cfrac%7B%5Ceta%7D%7B(%5Clambda_i%20-%20%5Cmu_j)%5E2%20%2B%20%5Ceta%5E2%7D%20u_i%20u_i%5E%5Ctop%20%5Cmathbf%7B1%7D%20%5Cmathbf%7B1%7D%5E%5Ctop%20v_j%20v_j%5E%5Ctop" alt="X = \sum_{i,j=1}^n \frac{\eta}{(\lambda_i - \mu_j)^2 + \eta^2} u_i u_i^\top \mathbf{1} \mathbf{1}^\top v_j v_j^\top">
  </p>
  
  where $\{(\lambda_i, u_i)\}$ are the eigenpairs of $A$, $\{(\mu_j, v_j)\}$ are the eigenpairs of $B$, and $\eta$ is a tuning hyperparameter.Then it solves a linear assignment problem of the similarity score $X$ to find the optimal correspondence
  
* **D-Hop**: this algorithm is different from the previous two as the D-Hop algorithm uses seeded information (i.e. a portion of the ground truth is known beforehand). Let $S$ denote the seed matrix, where $S_{ij}=1$ if we know the ground truth $i \leftrightarrow j$. The similarity matrix $X$ is computed as

To sum up, all of the statistical algorithms listed above are computationally fast, and they can also be categorized as unsupervised learning methods. The key difference is that the spectral methods such as Umeyama and pairwise spectral alignment do not leverage seeded information but the D-Hop method uses seeds to improve its performance.

### Graph Neural Network Method

## Experiment Setup
### Requirements
* Python
* PyTorch
* PyTorch Geometric
* Numpy
* Scipy

### Dataset Preparation
The Facebook network dataset is available at 

## Test Models on Synthetic Erdos-Renyi Graphs
The results are listed in the following table
| Number of Seeds             | 0% | 2% | 4% | 6% | 8% | 10% | 12% | 14% | 16% | 18% | 20% |
|-----------------------------|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| GM GNN                      |    |    |    |    |    |     |     |     |     |     |     |
| 2-Hop                       |    |    |    |    |    |     |     |     |     |     |     |
| 1-Hop                       |    |    |    |    |    |     |     |     |     |     |     |
| Pairwise Spectral Alignment |    |    |    |    |    |     |     |     |     |     |     |
| Umeyama                     |    |    |    |    |    |     |     |     |     |     |     |

## Test Models on Facebook Networks
The results are listed in the following table
| Number of Seeds             | 0% | 2% | 4% | 6% | 8% | 10% | 12% | 14% | 16% | 18% | 20% |
|-----------------------------|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| GM GNN                      |    |    |    |    |    |     |     |     |     |     |     |
| 2-Hop                       |    |    |    |    |    |     |     |     |     |     |     |
| 1-Hop                       |    |    |    |    |    |     |     |     |     |     |     |
| Pairwise Spectral Alignment |    |    |    |    |    |     |     |     |     |     |     |
| Umeyama                     |    |    |    |    |    |     |     |     |     |     |     |

## Test Models for De-anonymizing Twitter Network based on Facebook Network
