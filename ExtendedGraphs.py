from networkx.linalg import attrmatrix

# Defines a class for a signed/weighted directed graph.
# Note: please use this class for all the graphs in this project.
class SignedDirectedGraph:

  def __init__(self, graph):
    self.G = graph
    self.A, self.node_order = attrmatrix.attr_sparse_matrix(
        graph, edge_attr='weight')
    self.node_index = {}

    for idx, node in enumerate(self.node_order):
      self.node_index[node] = idx

    self.APos = self.A.maximum(0)
    self.ANeg = self.A.minimum(0).multiply(-1)
    self.APosT = self.APos.transpose()
    self.ANegT = self.ANeg.transpose()

  # Because the matrix is not indexed with node ID, so we need to
  # have this function to look up which position a node is actually
  # at in the matrix
  def lookUpNodeIdxInA(self, n):
    return self.node_index[n]

  def positiveAdjacencyMatrix(self):
    return self.APos

  def negativeAdjacencyMatrix(self):
    # obtain the negative enties of the matrix, and make them positive
    return self.ANeg

  def adjPermutations(self):
    return [self.APos, self.APosT, self.ANeg, self.ANegT]

