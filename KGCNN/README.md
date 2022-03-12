# Introduction To Keras Graph Convolutional Neural Network(KGCNN) & Ragged Tensor

Graph Neural Networks is a neural network architecture that has recently become more common in research publications and real-world applications.  And since neural graph networks require modified convolution and pooling operators, many Python packages like PyTorch Geometric, StellarGraph, and DGL have emerged for working with graphs.  In Keras Graph Convolutional Neural Network(kgcnn) a straightforward and flexible integration of graph operations into the TensorFlow-Keras framework is achieved using RaggedTensors. It contains a set of TensorFlow-Keras layer classes that can be used to build graph convolution models. The package also includes standard bench-mark graph datasets such as Cora,45 MUTAG46, and QM9.

The main problem with handling graphs is their variable size. This makes graph data hard to arrange in tensors. For example, placing small charts of different sizes in mini-batches poses a problem with fixed-sized tensors. One way to solve this problem is to use zero-padding with masking or composite tensors. Another is disjoint representation. This entails joining the small graphs into a single large graph without connecting the individual subgraphs. 

Graphs are usually represented by an adjacency matrix ???? of shape ([batch], N, N), which has ???????????? = 1if the graph has an edge between nodes i and j and 0 otherwise. When represented using Tensors, graphs are stored using:

- Node list n of shape ([batch], N, F)
- A connection table of edge indices of incoming and outgoing node m with shape([batch], M, 2)
- Corresponding edge feature list e of shape ([batch], M, F). 

Here, N denotes the number of nodes, F denotes the node representation dimension, and M the number of edges. 

RaggedTensors are the TensorFlow equivalent of nested variable-length lists. With RaggedTensors, graphs can be represented using just the node features and edge index lists a flexible tensor dimension that incorporates different numbers of nodes and edges. For example, a ragged node tensor of shape ([batch], None, F) can accommodate a flexible graph size in the second dimension.

A ragged tensor should not be confused with a sparse tensor, it is a dense tensor with an irregular shape.  The key difference is that a ragged tensor keeps track of where each row begins and ends, whereas a sparse tensor tracks each item’s coordinates. This difference can be illustrated using the concatenation operation:

1. https://analyticsindiamag.com/introduction-to-kgcnn-and-ragged-tensor/