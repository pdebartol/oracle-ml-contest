<img src="https://upload.wikimedia.org/wikipedia/commons/1/1d/Schziophrenia_PPI.jpg" height=250 width =300 align="right"></img>

# Abstract
Our task is to perform  <b>multi-label vertex classification</b> on the PPI  network to predict labels of unseen proteins and   generate a powerful embedding of the graph in a low dimensional space. The protein-protein-interaction network is represented with 24 different subgraphs ( we’ll use 20 for the training set and 4 for the test set ). Each one of our proteins has 50 features ( binary ), a list of  edges with other proteins and a label that represents its role ( if the protein belongs to the training set ). In order to build our embeddings  we’ll try to get the best information out  of connections, features and labels.  

## Key Issue
Coming up with a model that is able to capture graph structural information is really difficult. <b>There is  no straightforward way to encode this high-dimensional, non-Euclidean information into a feature vector</b>. In order to extract structural information from graphs, traditional approaches often rely on summary graph statistics or carefully engineered features to measure local neighborhood structures (pre processing). However, these approaches are limited because these hand-engineered features are inflexible, i.e. they cannot adapt during the learning process, and designing these features can be a time-consuming and expensive process.

## Recent Developments
The idea behind the latest approaches is to learn a mapping that embeds each node as a point in a low-dimensional vector space  <img src="https://latex.codecogs.com/gif.latex?\mathbb{R}^d" title="\mathbb{R}^d" />. Embedding should capture the graph topology, vertex-to-vertex relationship, and other relevant information about graphs, subgraphs, and vertices. Therefore, <b> this mapping is constructed carefully so that geometric relationships in the embedding space reflect the structure of the original graph</b>.In particular we assume that two nodes are neighbour in the embedding space if their features are similar, they belong to the same communities (homophily assumpion) and share similar roles (stuctural  assumption).  

# Our Approach
Throughout the repository :
* We'll use Node2Vec, GCN, SAGEConv and GAT   to highlight the importance of encoding graph structural information together with features in order to achieve a good vertex classifcation.
* We'll show that in this specific settings <b>the dataset is not good enough to create a good predictor</b> and that it is possibile to achieve the same performance with statistic methods ( not taking into account the graph and the feature )
* We must not forget that Machine Learning is not magic!

## Not enough?
Our thinking process is carefully explained in the main jupyter notebook, you should go take a look.
