# Defines the training framework for the signed link prediction task.
# Using primarily sklearn.

# Hao: given that our feature vectors are very large in total, since
# the number of edges in a graph is on the order of E5. We decide to 
# not use the sklearn.linear_model.LogisticRegression model as it does
# not support mini-bacthes and partial_fit.
# Instead, we choose to use sklearn.linear_model.SGDClassifier.

# Usage:
#  python main.py -d epinions -i 20
#  -d: datasetname
#  -i: number of iterations
import os
import sys
import getopt
import random
import time
import math
import networkx as nx
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from multiprocessing.dummy import Pool as ThreadPool

import highorder
from Node2vecExtractor import Node2vecExtractor
from featureExtraction import LoDegFeatureExtractor
from examples import getExampleGraph3


# This object configures which feature to use
featureConfig = {
  'l': False,
  'h': False,
  'n': False,
  'e': False
}

configString = ''

NUM_THREADS = 8

# a global highorder feature extractor
h_extractor = None
n_extractor = None
ld_extractor = None


def writeLog(dataset, logString):
  with open('result/{}/performance_{}.log'.format(
      dataset, configString), "a") as file:
    file.write(logString + '\n')


def loadGraph(dataset):
  # TODO: support more dataset.
  G = None
  edges = None
  if dataset == 'epinions':
    G = nx.read_weighted_edgelist(
        "soc-sign-epinions.txt",
        comments='#',
        create_using=nx.DiGraph(),
        encoding='utf-8')
  elif dataset == 'wiki':
    # TODO: implement loading
    G = nx.read_weighted_edgelist(
        'wiki.txt',
        comments='#',
        create_using=nx.DiGraph(),
        encoding='utf-8')
  elif dataset == 'G3':
    G = getExampleGraph3()

  if G is not None:
    edges = np.array(G.edges())

  return G, edges

# this type of indexing requires edges to be a numpy array
def sampleBatch(edges, batch_i):
  return edges[batch_i]


def makeBatches(batch_index, batchSize):
  random.shuffle(batch_index)
  batches = []
  idx = 0
  while idx < len(batch_index):
    batches.append(
        batch_index[idx:min(idx+batchSize, len(batch_index))])
    idx += batchSize

  return batches


# TODO(guo li): please follow the implementation of load_highorder
def load_degreetype(G, batch, X):
  res = ld_extractor.getDegreeFeatures(batch)
  # append features for every order, e.g 4, 5
  for i, row in enumerate(res):
    X[i].extend(row)


# TODO(leo): please follow the implementation of load_highorder
def load_loworder(G, batch, X):
  res = ld_extractor.getLowOrderFeatures(batch)
  # append features for every order, e.g 4, 5
  for i, row in enumerate(res):
    X[i].extend(row)


# This function add features to X, it doesn't return anything
def load_highorder(dataset, batch, X):
  global h_extractor

  res = h_extractor.getEdgeFeatures(batch)
  # append features for every order, e.g 4, 5
  for order in res:
    for i, row in enumerate(res[order]):
      X[i].extend(row)


def load_node2vec(dataset, batch, X):
  global n_extractor
  for i, edge in enumerate(batch):
    X[i].extend(n_extractor.getFeatureForEdge(edge[0], edge[1]))
  

# TODO: make this configurable so we can have different combinations
# of features.
def loadFeaturesSmallerBatch(graph, dataset, batch):
  global featureConfig

  batchSize = len(batch)
  X = [[] for i in xrange(batchSize)]
  if featureConfig['e']:
    load_degreetype(graph, batch, X)

  if featureConfig['l']:
    load_loworder(graph, batch, X)
  
  if featureConfig['h']:
    load_highorder(dataset, batch, X)
  
  if featureConfig['n']:
    load_node2vec(dataset, batch, X)

  return X

def loadFeatures(graph, dataset, batch):
  minibatches = []
  minibatchSize = len(batch) / NUM_THREADS
  for i in xrange(int(math.ceil(float(len(batch))/minibatchSize))):
    minibatches.append(batch[i*minibatchSize:min(len(batch), (i+1)*minibatchSize)])

  pool = ThreadPool(NUM_THREADS)
  results = pool.map(lambda mini: loadFeaturesSmallerBatch(graph, dataset, mini), minibatches)

  return reduce(lambda x, y: x + y, results)


def loadLabel(graph, batch):
  y = []
  for edge in batch:
    y.append(graph[edge[0]][edge[1]]['weight'])

  return y


def train(dataset, iters, batchSize):
  global ld_extractor
  # load the graph and get all the edges as an numpy array
  graph, edges = loadGraph(dataset)

  # build extractor only once.
  if featureConfig['l'] or featureConfig['e']:
    ld_extractor = LoDegFeatureExtractor(graph)

  # os.makedirs('result/{}'.format(dataset))

  # Use cross validation
  rocScoreOverall = 0.0
  accScoreOverall = 0.0
  kf = KFold(n_splits=10, shuffle = True)
  # partition the data to:
  #   -10% validation/test
  #   -90% training

  for i, (train_index, test_index) in enumerate(kf.split(edges)):
    # logistic regression with l2 regularization
    clf = SGDClassifier(loss='log', l1_ratio=0)
    for it in xrange(iters):
      # generate randomized training batches
      
      trainBatches = makeBatches(train_index, batchSize)
      validationBatches = makeBatches(test_index, batchSize)

      for k_idx, batch_i in enumerate(trainBatches):
        # sample a mini-batch of size batchSize
        # the batch is a list of edges.
        print k_idx
        batch = sampleBatch(edges, batch_i)

        # load features to X of shape (batchSize, f)
        X = loadFeatures(graph, dataset, batch)

        # load labels for this batch
        # y is numpy array, shape (n_samples,)
        y = loadLabel(graph, batch)

        clf.partial_fit(X, y, classes=[-1, 1])

    # time to varify our performance
    rocScore = 0.0
    accScore = 0.0
    for batch_i in validationBatches:
      batch = sampleBatch(edges, batch_i)

      X = loadFeatures(graph, dataset, batch)

      y = loadLabel(graph, batch)

      yPred = clf.predict(X)
      # yPred is an array of 1 and -1

      rocScore += roc_auc_score(y, yPred)
      accScore += accuracy_score(y, yPred)

    # get the average rocScore for all the batches
    rocScore /= len(validationBatches)
    accScore /= len(validationBatches)

    logString = 'For validation fold #{}:\n roc is {}\n accuracy is {}'.format(
          i, rocScore, accScore)
    print logString
    writeLog(dataset, logString)

    rocScoreOverall += rocScore
    accScoreOverall += accScore

  rocScoreOverall /= 10.0
  accScoreOverall /= 10.0
  summary = 'Overall, roc is: {}, accuracy is {}'.format(
      rocScoreOverall, accScoreOverall)
  print summary
  writeLog(dataset, summary)


if __name__ == '__main__':
  optval, leftover = getopt.getopt(sys.argv[1:], 'd:i:b:lhne')
  # default dataset is G3
  dataset = 'G3'
  # default iteration is 20
  iters = 20
  # default batch size is 64
  batchSize = 64

  for o, v in optval:
    if o == '-d':
      dataset = v
    elif o == '-i':
      iters = int(v)
    elif o == '-b':
      batchSize = int(v)
    elif o == '-l':
      featureConfig['l'] = True
      configString += 'l'
    elif o == '-h':
      featureConfig['h'] = True
      configString += 'h'
    elif o == '-n':
      featureConfig['n'] = True
      configString += 'n'
    elif o == '-e':
      featureConfig['e'] = True
      configString += 'e'
    else:
      pass

  print 'Start training on {} dataset, {} iterations total'.format(
      dataset, iters)

  # create a global high order feature extractor for this dataset.
  if featureConfig['h']:
    h_extractor = highorder.HighOrderFeatureExtractor(dataset, [4, 5])
  if featureConfig['n']:
    n_extractor = Node2vecExtractor(dataset)
    print 'built n extractor'
  
  train(dataset, iters, batchSize)
