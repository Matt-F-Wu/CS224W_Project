def getNodeNbrs(G, nodeId):
  node = G.GetNI(nodeId)
  nodedegree = node.GetDeg()
  nbrIds = []
  for i in range(nodedegree):
    nbrIds.append(node.GetNbrNId(i))

  return nbrIds

def getNodeInNbrs(G, nodeId):
  node = G.GetNI(nodeId)
  nodedegree = GetInDeg()
  nbrIds = []
  for i in range(nodedegree):
    nbrIds.append(node.GetInNId(i))

  return nbrIds

def getNodeOutNbrs(G, nodeId):
  node = G.GetNI(nodeId)
  nodedegree = node.GetOutDeg()
  nbrIds = []
  for i in range(nodedegree):
    nbrIds.append(node.GetOutNId(i))

  return nbrIds
