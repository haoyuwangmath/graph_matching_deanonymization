# Social Network De-anonymization via Graph Matching
This repository discusses graph matching algorithms for the social network de-anonymization problem, which refers to finding the same users between different social networks that are possibly anonymized. Compared with other previous works that leverages user profile (e.g. user id), our approaches only rely on graph topological information. 

We will focus on two types of algorithms: traditional statistical approaches and a graph neural network approach.

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

  <p align="center">
  <img src="https://latex.codecogs.com/png.latex?X%20%3D%20A_%7B(D)%7D%20S%20B_%7B(D)%7D%5E%5Ctop%2C%20%5Cquad%20A_%7B(D)%7D%20%3D%20%5B(((A_%7B(D)%7D%20A)%20%3E%200)%20-%20%5Csum_%7Bk%3D1%7D%5E%7BD-1%7D%20A_%7B(k)%7D%20-%20I)%20%3E%200%5D%2C%5Cquad%20B_%7B(D)%7D%20%3D%20%5B(((B_%7B(D)%7D%20B)%20%3E%200)%20-%20%5Csum_%7Bk%3D1%7D%5E%7BD-1%7D%20B_%7B(k)%7D%20-%20I)%20%3E%200%5D" alt="X = A_{(D)} S B_{(D)}^\top, \quad A_{(D)} = [(((A_{(D)} A) > 0) - \sum_{k=1}^{D-1} A_{(k)} - I) > 0],\quad B_{(D)} = [(((B_{(D)} B) > 0) - \sum_{k=1}^{D-1} B_{(k)} - I) > 0]">
  </p>
  
To sum up, all of the statistical algorithms listed above are computationally fast, and they can also be categorized as unsupervised learning methods. The key difference is that the spectral methods such as Umeyama and pairwise spectral alignment do not leverage seeded information but the D-Hop method uses seeds to improve its performance.

### Graph Neural Network Method
As mentioned above, statistical algorithms are unsupervised. A natural question is if there is a supervised learning approach for graph matching. This line of work focuses on graph neural networks. In particular, the GNN we use here also leverages some insights of statistical methods.

## Experiment Setup
### Requirements
* Python (>=3.8)
* PyTorch (>=1.2.0)
* PyTorch Geometric (>=1.5.0)
* Numpy (>=1.20.1)
* Scipy (>=1.6.2)

### Training Data
The Facebook network dataset for GNN training is available at [here](https://archive.org/download/oxford-2005-facebook-matrix/facebook100.zip).

## Test on Synthetic Erdos-Renyi Graphs
The sparsity of synthetic graphs is a significant parameter. We consider both the dense regime $p=0.3$ and the sparse regime $p=0.01$. In both regime, we set the correlation parameter to be $s=0.8$

For sparse $s$-correlated Erdos-Renyi graphs $G(n,p,s)$ with $p=0.01$, the accuracy(%) of various algorithms are listed in the following table
| Number of Seeds             | 0% | 2% | 4% | 6% | 8% | 10% | 12% | 14% | 16% | 18% | 20% |
|-----------------------------|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| GM GNN                      |    |    |    |    |    |     |     |     |     |     |     |
| 2-Hop                       |    |    |    |    |    |     |     |     |     |     |     |
| 1-Hop                       |    |    |    |    |    |     |     |     |     |     |     |


For dense $s$-correlated Erdos-Renyi graphs $G(n,p,s)$ with $p=0.3$, the results are listed in the following table
| Number of Seeds             | 0% | 2% | 4% | 6% | 8% | 10% | 12% | 14% | 16% | 18% | 20% |
|-----------------------------|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| GM GNN                      |  0.2  |  0.5  |  12.2  |  88.1  |  90.3  |  97.4   |  100   |  100   |  100   |   100  |  100   |
| 2-Hop                       |  0.1  |  0.1  |  2.2   |  6.6   |  40.7  |  100    |  100   |  100   |  100   |  100   |   100  |
| 1-Hop                       |  0.1  |  0.3  |  3.3   |  7.4   |  90.6  |  100    |  100   |  100   |  100   |   100  |  100   |


## Test on Facebook Networks
The test results for Facebook networks are listed in the following table
| Number of Seeds             | 0% | 2% | 4% | 6% | 8% | 10% | 12% | 14% | 16% | 18% | 20% |
|-----------------------------|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| GM GNN                      |    |    |    |    |    |     |     |     |     |     |     |
| 2-Hop                       |    |    |    |    |    |     |     |     |     |     |     |
| 1-Hop                       |    |    |    |    |    |     |     |     |     |     |     |
| Pairwise Spectral Alignment |    |    |    |    |    |     |     |     |     |     |     |
| Umeyama                     |    |    |    |    |    |     |     |     |     |     |     |


## Test for de-anonymizing Twitter-Flickr Networks
We compare our algorithms with the existing work.
