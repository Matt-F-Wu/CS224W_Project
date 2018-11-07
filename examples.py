import networkx as nx
from ExtendedGraphs import SignedDirectedGraph

def getExampleGraph1():
  DG = nx.DiGraph()

  for i in range(1, 9):
    DG.add_node(i)

  DG.add_edge(1, 2, weight=1)
  DG.add_edge(2, 3, weight=-1)
  DG.add_edge(3, 1, weight=1)
  DG.add_edge(2, 4, weight=1)
  DG.add_edge(3, 4, weight=-1)
  DG.add_edge(4, 5, weight=-1)
  DG.add_edge(5, 1, weight=1)
  DG.add_edge(5, 6, weight=1)
  DG.add_edge(6, 7, weight=-1)
  DG.add_edge(6, 8, weight=-1)
  DG.add_edge(7, 8, weight=1)

  return DG

def getExampleGraph2():
  DG = nx.DiGraph()

  for i in range(1, 10):
    DG.add_node(i)

  DG.add_edge(1, 2, weight=1)
  DG.add_edge(1, 4, weight=1)
  DG.add_edge(1, 8, weight=1)
  DG.add_edge(2, 4, weight=-1)
  DG.add_edge(3, 4, weight=-1)
  DG.add_edge(3, 5, weight=1)
  DG.add_edge(8, 5, weight=-1)
  DG.add_edge(9, 8, weight=1)
  DG.add_edge(9, 6, weight=-1)
  DG.add_edge(5, 6, weight=1)
  DG.add_edge(5, 7, weight=1)
  DG.add_edge(6, 7, weight=-1)

  return DG

def getExampleGraph3():
  DG = nx.DiGraph()

  for i in range(1, 60):
    DG.add_node(i)

  w = 1
  for i in range(1, 59):
    w *= -1 
    DG.add_edge(i, i+1, weight=w)

  return DG

if __name__ == '__main__':
  G = getExampleGraph2()
