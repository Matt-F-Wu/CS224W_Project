import examples
# This file contains method used to extract long cycle/walk features,
# and other hight-order features

# Expect G to be of type SignedDirectedGraph
def adjacencyMatrixBranchMultiply(prod, G, k, i, j, res):
  if k == 0:
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
  # Trigger recursive call, this is very expensive.
  adjacencyMatrixBranchMultiply(None, G, k - 1, i, j, res)

  return res

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

