import snap
from scipy import sparse

from utils import getNodeInNbrs, getNodeOutNbrs

# Defines a class for a signed/weighted directed graph.
# Note: please use this class for all the graphs in this project.
class SignedDirectedGraph:

  def __init__(self, *args, **kwargs):
    self.G = snap.PNGraph.New()
    self.signMap = {}
    self.A = None

  # Add a signed edge, the sign could either be 1 or -1
  def AddSignedEdge(self, SrcNId, DstNId, sign):
    res = self.G.AddEdge(SrcNId, DstNId)
    # Store the sign in a dictionary
    if res == -1:
      # edge is successfully added
      self.signMap[(SrcNId, DstNId)] = sign

    return res

  # This function should only be called once after graph is loaded,
  # or when the graph structure is changed, e.g adding/removing an edge.
  # Reference for lil_matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html#scipy.sparse.lil_matrix
  def computeAdjacencyMatrix(self):
    max_id = None
    for node in self.G.Nodes():
      if max_id is None or node.GetId() > max_id:
        max_id = node.GetId()

    # Define the size for the adjacency matrix.
    n = max_id + 1
    # Use scipy's sparse matrix implementation to save memory
    # also use lil_matrix to support fancy indexing etc.
    self.A = sparse.lil_matrix((n, n))

    for edge in self.G.Edges():
      self.A[edge.GetSrcNId(), edge.GetDstNId()] = self.signMap[(edge.GetSrcNId(), edge.GetDstNId())]

  def positiveAdjacencyMatrix(self):
    return self.A.maximum(0)

  def negativeAdjacencyMatrix(self):
    # obtain the negative enties of the matrix, and make them positive
    return self.A.minimum(0).multiply(-1)

  def AddNode(self, id):
    return self.G.AddNode(id)

  def AddEdge(self, src, dest):
    return self.G.AddEdge(src, dest)

  def Nodes(self):
    return self.G.Nodes()

  def Edges(self):
    return self.G.Edges()

  def GetNodes(self):
    return self.G.GetNodes()

  def GetEdges(self):
    return self.G.GetEdges()

