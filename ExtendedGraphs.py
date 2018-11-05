from networkx.linalg import attrmatrix
from scipy import sparse

# Defines a class for a signed/weighted directed graph.
# Note: please use this class for all the graphs in this project.
class SignedDirectedGraph:

  def __init__(self, graph):
    self.G = graph
    self.A, self.node_order = attrmatrix.attr_sparse_matrix(
        graph, edge_attr='weight')

  # Because the matrix is not indexed with node ID, so we need to
  # have this function to look up which position a node is actually
  # at in the matrix
  def lookUpNodeIdxInA(self, n):
    return self.node_order.index(n)

  def positiveAdjacencyMatrix(self):
    return self.A.maximum(0)

  def negativeAdjacencyMatrix(self):
    # obtain the negative enties of the matrix, and make them positive
    return self.A.minimum(0).multiply(-1)

