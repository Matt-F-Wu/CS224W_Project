import os
import time
import pickle
import uuid
import gc
import numpy as np
from scipy import sparse
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool

import examples
from ExtendedGraphs import SignedDirectedGraph

# This file contains method used to extract long cycle/walk features,
# and other hight-order features

# Hao: this particular threading configuration is chosen to optimize
# performance on machines with #CPU > 8 and RAM >= 512GB.
NUM_BRANCHING_THREADS = 4

def save_obj(obj, name):
  # print type(obj)
  with open('obj/'+ name + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
  with open('obj/' + name + '.pkl', 'rb') as f:
    return pickle.load(f)

def save_npz(sparse_mtx, name):
  sparse.save_npz('obj/' + name + '.npz', sparse_mtx)

def load_npz(name):
  return sparse.load_npz('obj/' + name + '.npz')

# Expect G to be of type SignedDirectedGraph
def adjacencyMatrixBranchMultiply(prod, G, k, res, name,
    precompute=False):
  if precompute:
    # A^2(s) have not been precomputed.
    if k < 2:
      # Write result to file, and add file name to res.
      filename = '{}/{}'.format(name, uuid.uuid4())
      save_npz(prod, filename)
      res.append(filename)
      return

    for mtx in G.adjPermutations():
      if prod is None:
        adjacencyMatrixBranchMultiply(
            mtx, G, k - 1, res, name, precompute)
      else:
        adjacencyMatrixBranchMultiply(
            prod.dot(mtx), G, k - 1, res, name, precompute)
    return

  if k > 2:
    # We could utilize the A^2 matrices already stored
    filenames = load_obj('{}/hf_A2'.format(name))
    for filename in filenames:
      mtx = load_npz(filename)
      if prod is None:
        adjacencyMatrixBranchMultiply(mtx, G, k - 2, res, name)
      else:
        adjacencyMatrixBranchMultiply(
            prod.dot(mtx), G, k - 2, res, name)
  elif k == 2:
    # Multiply with A
    for mtx in G.adjPermutations():
      if prod is None:
        adjacencyMatrixBranchMultiply(mtx, G, k - 1, res, name)
      else:
        adjacencyMatrixBranchMultiply(
            prod.dot(mtx), G, k - 1, res, name)
  else:
    # Write result to file, and add file name to res.
    filename = '{}/{}'.format(name, uuid.uuid4())
    save_npz(prod, filename)
    res.append(filename)
  
  return

def longWalkFeature(G, k_list, name, precompute=False):
  resAll = defaultdict(list)
  
  for k in k_list:
    res = []
    adjacencyMatrixBranchMultiply(None, G, k, res, name, precompute)
    resAll[k] = res

  return resAll

def longWalkFeatureWriteAll(G, k_list, name):
  # create a subdirectory under obj to store all data relevant to this
  # graph.
  os.mkdir('obj/{}'.format(name))
  G = SignedDirectedGraph(G)
  save_obj(G.node_index, '{}/node_index'.format(name))
  
  # Preprocessing step:
  # Store all the A^2 results, these results will be used to
  # consititute higher powers to eliminate duplicate computation.
  A2res = longWalkFeature(G, [3], name, precompute=True)
  # Path of length 3 computes A^2.
  save_obj(A2res[3], '{}/hf_A2'.format(name))

  # Actually computing the targeted lengths.
  res = longWalkFeature(G, k_list, name)
  
  for k in k_list:
    save_obj(res[k], '{}/hf_{}'.format(name, k))

def longWalkFeatureExtract(feature, node_index, i, j):
  i = node_index[i]
  j = node_index[j]

  return [fe[i, j] for fe in feature]

# helper function to read multiple entries from a sparse matrix
# stored in a file. The number of entries to read needs to be
# carefully selected in order to balance out file I/O cost vs memory # consumption.
def readEntries(filename, i, j):
  sparse_mtx = load_npz(filename)
  # print type(sparse_mtx)
  return sparse_mtx[i, j];

# This class is constructed from loading the file references for
# sparse matrices corresponding to cycles counts of different length,
# for a specific graph, the name arg identifies the graph.
#
# It also loads the dictionary that mapps node number to mtx index.
# Note:
#   This class should be constructed once to obtain the handlers.
class HighOrderFeatureExtractor(object):
  """docstring for HighOrderFeatureExtractor"""
    
  def __init__(self, name, k_list):
    super(HighOrderFeatureExtractor, self).__init__()
    node_index = load_obj('{}/node_index'.format(name))
    feature_store = {}
    for k in k_list:
      feature_store[k] = load_obj('{}/hf_{}'.format(name, k))

    self.node_index = node_index
    self.feature_store = feature_store

  # Get the high order features for one or a list of edges
  # Arg:
  #  node_index: dictionary, key is node id and value is index in mtx.
  #  feature_store: dictionary, key is order number, valus is a list
  #    files storing various configurations of mtx representing various
  #    path configurations.
  #  edge: tuple or list of tuples
  def getEdgeFeatures(self, edges):
    # since the feature is really a bunch of filenames storing sparse matrices, we need to read the features in batch.
    node_index = self.node_index
    feature_store = self.feature_store

    # The following 2 variables are intended to be lists
    src = None
    dst = None
    if type(edges) is tuple:
      src = [edges[0]]
      dst = [edges[1]]
    elif type(edges) is list:
      # get an array of src nodes, and an arry of dst nodes
      src, dst = zip(*edges)
    else:
      # argument type might be numpy array
      src = []
      dst = []
      for edge in edges:
        src.append(edge[0])
        dst.append(edge[1])

    # Get lists of mtx indices for src and dst nodes
    i = [node_index[s] for s in src]
    j = [node_index[d] for d in dst]

    # edgeFeatures is a dictionary of (list of lists)
    # 1. the dict has keys corresponding to orders: e.g 4, 5
    # 2. the value of the dict are a list of length equal to that of 
    #   edge
    # 3. each element in that list is a feature vector for the
    #   corresponding edge for a specific order.
    edgeFeatures = {}
    pool = ThreadPool(NUM_BRANCHING_THREADS * 8)
    for order in feature_store:
      edgeFeatures[order] = np.zeros((len(i), 4**(order - 1)))
      # use multithreading to speed up loading
      def featureAppend(arg):
        idx, filename = arg
        edgeFeatures[order][:, idx] = readEntries(filename, i, j)

      pool.map(featureAppend, enumerate(feature_store[order]))
    pool.close()
    pool.terminate()
    # Row n of edgeFeature[order] represent a feature vector of that
    # order for the n-th edge in edges.
    return edgeFeatures

if __name__ == '__main__':  
  # Test computation and saving object
  G3 = examples.getExampleGraph3()
  start = time.time()
  longWalkFeatureWriteAll(G3, [4, 5], 'G3')
  end = time.time()

  print 'Runtime: {:.2f} minutes'.format((end - start) / 60)
  
  # Test loading object
  h_extractor = HighOrderFeatureExtractor('G3', [4, 5])
  res3 = h_extractor.getEdgeFeatures([(1, 2), (2, 3)])
  print sum(res3[4][0]) # expects 2.0
  print sum(res3[4][1]) # expects 3.0