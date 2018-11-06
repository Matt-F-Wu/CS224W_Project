from collections import defaultdict
import examples
# This file contains method used to extract long cycle/walk features,
# and other hight-order features

# Expect G to be of type SignedDirectedGraph
def adjacencyMatrixBranchMultiply(prod, G, k, i, j, res):
  if k == 0:
    if i is None and j is None:
      # we are writing everything to file
      if res is None:
        pass
      else:
        # we are storing features for all pairs of nodes in a dictionary.
        row, column = prod.get_shape()
        for x in xrange(row):
          for y in xrange(column):
            res[(G.node_order[x], G.node_order[y])].append(prod[x, y])

      return

    res.append(prod[i, j])
    return

  if prod is None:
    adjacencyMatrixBranchMultiply(G.positiveAdjacencyMatrix(), G, k-1, i, j, res)
    adjacencyMatrixBranchMultiply(G.positiveAdjacencyMatrix().transpose(), G, k-1, i, j, res)
    adjacencyMatrixBranchMultiply(G.negativeAdjacencyMatrix(), G, k-1, i, j, res)
    adjacencyMatrixBranchMultiply(G.negativeAdjacencyMatrix().transpose(), G, k-1, i, j, res)
  else:
    adjacencyMatrixBranchMultiply(prod.dot(G.positiveAdjacencyMatrix()), G, k-1, i, j, res)
    adjacencyMatrixBranchMultiply(
        prod.dot(G.positiveAdjacencyMatrix().transpose()), G, k-1, i, j, res)
    adjacencyMatrixBranchMultiply(prod.dot(G.negativeAdjacencyMatrix()), G, k-1, i, j, res)
    adjacencyMatrixBranchMultiply(
        prod.dot(G.negativeAdjacencyMatrix().transpose()), G, k-1, i, j, res)

def longWalkFeature(G, k, i, j):
  res = []
  # identify the index of node i and j in the adjacency matrix.
  i = G.lookUpNodeIdxInA(i)
  j = G.lookUpNodeIdxInA(j)

  # Trigger recursive call, this is very expensive.
  adjacencyMatrixBranchMultiply(None, G, k - 1, i, j, res)

  return res

def longWalkFeatureAll(G, k_list):
  all_res = []
  for k in k_list:
    res = defaultdict(list)
    adjacencyMatrixBranchMultiply(None, G, k - 1, None, None, res)
    all_res.append(res)

  return all_res

def longWalkFeatureWriteAll(G, k_list):
  for k in k_list:
    adjacencyMatrixBranchMultiply(None, G, k - 1)

if __name__ == '__main__':  
  G1 = examples.getExampleGraph1()
  res1 = longWalkFeature(G1, 4, 1, 5)
  print sum(res1)

  G2 = examples.getExampleGraph2()
  res2 = longWalkFeature(G2, 4, 5, 8)
  print sum(res2)
  
  G3 = examples.getExampleGraph3()
  res3 = longWalkFeature(G3, 4, 1, 2)
  print sum(res3)

  g3_all_h_features = longWalkFeatureAll(G3, [4, 5])
  print g3_all_h_features
