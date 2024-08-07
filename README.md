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

The Gm-GNN model operates on pairs of nodes across two graphs, allowing it to effectively utilize seed information and compute witness-like information from different hops. The architecture comprises two main modules: the Convolution Module and the Percolation Module. The Convolution Module aggregates neighborhood information to update the similarity between node pairs, capturing witness-like information from various hops. This aggregation enables GM-GNN to generalize to graphs of different sizes and structures. The Percolation Module, on the other hand, filters out low-similarity node pairs and uses high-confidence pairs as new seeds for subsequent layers, thereby enhancing the matching accuracy over multiple iterations.

## Experiment Setup
### Requirements
* Python (>=3.8)
* PyTorch (>=1.2.0)
* PyTorch Geometric (>=1.5.0)
* Numpy (>=1.20.1)
* Scipy (>=1.6.2)

### Training Data
The Facebook network dataset for GNN training is available at [here](https://archive.org/download/oxford-2005-facebook-matrix/facebook100.zip).


## Training of Models
The training dataset of our GNN has two components: synthetic Erdos-Renyi graphs and Facebook networks. We will set the fraction of seeds to be $\theta=0.1$ in the training process. For the synthetic Erdos-Renyi graphs, we set parameters $n=500$, $p\in\\{0.02,0.1,0.3,0.5\\}$ and $s\in\\{0.4,0.6,0.8,1.0\\}$. For each combination of parameters, we generate 10 independent pairs of correlated Erdos-Renyi graphs. For the Facebook networks, due to limitation of computational resources, we will randomly subsample $n=500$ nodes in each network and add the induced subgraphs into our training set.


## Test on Synthetic Erdos-Renyi Graphs
### Seeded Matching
The sparsity of synthetic graphs is a significant parameter. We consider both the dense regime $p=0.3$ and the sparse regime $p=0.01$. In both regime, we set the number of nodes to be $n=1000$ and the correlation parameter to be $s=0.8$

For sparse $s$-correlated Erdos-Renyi graphs $G(n,p,s)$ with $p=0.01$, the accuracy(%) of various algorithms are listed in the following table
| Number of Seeds             | 0% | 2% | 4% | 6% | 8% | 10% | 12% | 14% | 16% | 18% | 20% |
|-----------------------------|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| GM GNN                      |  0.2  |  12.4  |  38.8  | 76.9   |  85.1  |  95.8   |  96.0   |  96.2   |  96.6   |  97.4   |  99.8   |
| 2-Hop                       |  0.2  |  2.4  |  16.2  |  49.2  | 77.1   |  83.4   |  88.8   |  92.3   |  94.4   |   95.8  |   96.5  |
| 1-Hop                       |  0.3  |  1.1  |  2.6  |  4.7  |  6.0  |   9.8  |  11.8   |  16.4   |  22.0   |   32.6  |  40.3   |


For dense $s$-correlated Erdos-Renyi graphs $G(n,p,s)$ with $p=0.3$, the results are listed in the following table
| Number of Seeds             | 0.0% | 0.5% | 1.0% | 1.5% | 2.0% | 2.5% | 3.0% | 3.5% | 4.0% | 4.5% | 5.0% |
|-----------------------------|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| GM GNN                      |  0.2  |  0.5  |  12.2  |  88.1  |  90.3  |  97.4   |  100   |  100   |  100   |   100  |  100   |
| 2-Hop                       |  0.1  |  0.1  |  2.2   |  6.6   |  40.7  |  100    |  100   |  100   |  100   |  100   |   100  |
| 1-Hop                       |  0.1  |  0.3  |  3.3   |  7.4   |  90.6  |  100    |  100   |  100   |  100   |   100  |  100   |

### Seedless Matching
For seedless graph matching, we can use statistical algorithms to generate an initial matching as partially correct seeds and use our GNN to refine it. Again we consider both the dense regime $p=0.3$ and the sparse regime $p=0.01$. We test the algorithms on varying graph correlations and compare our algorithms with existing GNN methods (e.g. [GMN](https://github.com/stones-zl/PCA-GM), [DGMC](https://github.com/rusty1s/deep-graph-matching-consensus) ) for seedless matching.

For the dense graphs:
| Graph Correlation             | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|-------------------------------|------|------|------|------|------|------|------|------|------|
| GM GNN + Pairwise Spectral Alignment   |   13.1    |   25.6    |   77.4    |    90.2   |    97.8   |   100    |   100    |   100    |    100   |
| GM GNN + Umeyama                       |   9.8    |   18.8    |   64.2    |    74.7   |    78.5   |   80.8    |   86.4    |   100    |    100   |
| DGMC                                   |   14.8    |   22.3    |   78.6    |    84.4   |    90.2   |   94.7    |   100    |   100    |    100   |
| GMN                                    |   7.5    |   25.6    |   60.7    |    66.5   |   68.1   |   72.9    |   92.1    |   100    |    100   |


For the sparse graphs:
| Graph Correlation             | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|-------------------------------|------|------|------|------|------|------|------|------|------|
| GM GNN +  Pairwise Spectral Alignment   |  8.4    |  12.4    |   40.9   |   65.1   |  80.8    |  85.3    |  90.4    |  94.1    |   96.7   |
| GM GNN + Umeyama                        |   4.1   |   7.7   |   11.8   |   42.6   |   70.4   |   86.1   |   89.6   |   91.6   |    92.4  |
| DGMC                                    |   6.2   |   10.6   |   34.7   |   64.9   |  81.2    |  84.9    |   92.4   |  93.1    |   94.1   |
| GMN                                     |   2.8   |   3.2   |   9.9   |  20.7    |   46.5   |   61.7   |   70.8   |   77.3   |   80.2   |


## Test on Facebook Networks
The test results for Facebook networks are listed in the following table
| Number of Seeds             | 0% | 1% | 2% | 3% | 4% | 5% | 6% | 7% | 8% | 9% | 10% |
|-----------------------------|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| GM GNN                      |  1.4  |  16.7  |  78.1  |  82.9  |  84.0  |  84.4   |  88.1   |  90.4   |  91.2   |  91.6   |  92.3   |
| 2-Hop                       |  1.8  |  9.7  |  47.2  |  68.8  |  80.2  |  82.7   |  84.1   |  86.3   |  87.2   |  87.4   |   89.1  |
| 1-Hop                       |  1.2  |  4.6  |  33.6  |  57.4  |  78.0  |  83.6   |  83.9   |  86.4   |  86.6   |  87.7   |  88.8   |

For seedless matching, the results are
| Graph Correlation             | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|-------------------------------|------|------|------|------|------|------|------|------|------|
| GM GNN +  Pairwise Spectral Alignment   |  6.4    |  11.3    |  18.8    |  60.9    |  64.0    |  70.3    |  72.6    |  80.2    |   84.3   |
| GM GNN + Umeyama                        |  2.9    |  4.1     |  6.8     |  48.4    |  50.9    |  62.1    |  68.8    |  72.1    |   78.6   |
| DGMC                                    |  4.4    |  10.2    |  15.7    |  55.4    |  62.2    |  64.3    |  66.5    |  77.6    |   80.8   |
| GMN                                     |  1.3    |  2.6     |  4.8     |  23.8    |  30.4    |  48.7    |  50.1    |  54.4    |   66.9   |



## Test for Twitter-Flickr Networks
We compare our algorithms with the existing work ([link](https://snap.stanford.edu/class/cs224w-2012/projects/cs224w-053-final.pdf)).
| GM GNN + Pairwise Spectral Alignment     |     GM GNN + Umeyama | Greedy + Overlap* | ExactMatch* |
|------------------------------------------|----------------------|-------------------|-------------|
|              71.42%                      |        64.05%        |    59.31%         |    39.76%   |
