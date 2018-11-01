from ExtendedGraphs import SignedDirectedGraph

def getExampleGraph1():
  G = SignedDirectedGraph()

  for i in range(1, 9):
    G.AddNode(i)

  print type(G)

  G.AddSignedEdge(1, 2, 1)
  G.AddSignedEdge(2, 3, -1)
  G.AddSignedEdge(3, 1, 1)
  G.AddSignedEdge(2, 4, 1)
  G.AddSignedEdge(3, 4, -1)
  G.AddSignedEdge(4, 5, -1)
  G.AddSignedEdge(5, 1, 1)
  G.AddSignedEdge(5, 6, 1)
  G.AddSignedEdge(6, 7, -1)
  G.AddSignedEdge(6, 8, -1)
  G.AddSignedEdge(7, 8, 1)

  # This call is necessary
  G.computeAdjacencyMatrix()

  return G

def getExampleGraph2():
  G = SignedDirectedGraph()

  for i in range(1, 10):
    G.AddNode(i)

  G.AddSignedEdge(1, 2, 1)
  G.AddSignedEdge(1, 4, 1)
  G.AddSignedEdge(1, 8, 1)
  G.AddSignedEdge(2, 4, -1)
  G.AddSignedEdge(3, 4, -1)
  G.AddSignedEdge(3, 5, 1)
  G.AddSignedEdge(8, 5, -1)
  G.AddSignedEdge(9, 8, 1)
  G.AddSignedEdge(9, 6, -1)
  G.AddSignedEdge(5, 6, 1)
  G.AddSignedEdge(5, 7, 1)
  G.AddSignedEdge(6, 7, -1)

  G.computeAdjacencyMatrix()

  return G

def getExampleGraph3():
  G = SignedDirectedGraph()

  for i in range(1, 6):
    G.AddNode(i)

  G.AddSignedEdge(1, 2, 1)
  G.AddSignedEdge(2, 3, 1)
  G.AddSignedEdge(3, 4, 1)
  G.AddSignedEdge(4, 5, 1)

  G.computeAdjacencyMatrix()

  return G
