#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:56:03 2018

@author: liguo
"""
import networkx as nx
from random import choice
import examples as ex
import numpy as np

import highorder


# Degree type feature and lowOrder feature:
def degreeFeature(graph, srcNode, dstNode): # TUNGraphEdgeI
    # get source and destination node id for given edge in given graph
    #srcNode, dstNode, weight= edge
    weight = graph[srcNode][dstNode]['weight']
    featureList = []
    print "Nodes read!: ", srcNode, dstNode, weight
    
    # out degree for source node
    posOutDeg_src = graph.out_degree(srcNode, weight = 1)
    negOutDeg_src = graph.out_degree(srcNode, weight = -1)
    featureList.append(posOutDeg_src)
    featureList.append(negOutDeg_src)
    print "Out degree for source node", posOutDeg_src, negOutDeg_src
    
    # in degree for destination node
    posInDeg_dst = graph.out_degree(dstNode, weight = 1)
    negInDeg_dst = graph.out_degree(dstNode, weight = -1)
    featureList.append(posInDeg_dst)
    featureList.append(negInDeg_dst)
    print "in degree for destination node", posInDeg_dst, negInDeg_dst
    
    # total out degree for start node
    totalOut_src = posOutDeg_src + negOutDeg_src
    featureList.append(totalOut_src);
    print "total out degree for start node", totalOut_src
    
    # total in degree for start node
    totalIn_dst = posInDeg_dst + negInDeg_dst
    featureList.append(totalIn_dst);
    print "total in degree for start node", totalIn_dst
    
    # change graph into undirected graph
    uGraph = graph.to_undirected()
    
    # number of common neighbors
    cmnNbrs = list(nx.common_neighbors(uGraph, srcNode, dstNode))
    cmnNbrNum = len(cmnNbrs)
    featureList.append(cmnNbrNum)
    print "number of common neighbors", cmnNbrNum
    
    # neighbors(u) * neighbors(v)
    nbr_src = list(uGraph.neighbors(srcNode))
    nbr_dst = list(uGraph.neighbors(dstNode))
    nbr_srcNum = len(nbr_src)
    nbr_dstNum = len(nbr_dst)
    featureList.append(nbr_srcNum * nbr_dstNum)
    print "neighbors(u) * neighbors(v)", nbr_srcNum * nbr_dstNum
    
    # Jaccard Coefficient for given pair of nodes
    triple = nx.jaccard_coefficient(uGraph, [(srcNode, dstNode)])
    for u, v, jaccardCoef in triple:
        featureList.append(jaccardCoef)
    print "Jaccard Coefficient for given pair of nodes", jaccardCoef

    return featureList


            
# Extract all triads types with given edge inside of them
def lowOrderFeature(graph, srcNode, dstNode):
    # change graph into undirected graph
    uGraph = graph.to_undirected()
    
    # number of common neighbors
    cmnNbrs = list(nx.common_neighbors(uGraph, srcNode, dstNode))
    motif_counts = [0] * 16
    for nbr in cmnNbrs:
        edge1 = graph.get_edge_data(srcNode, nbr, default=0)
        edge2 = graph.get_edge_data(nbr, srcNode, default=0)
        edge3 = graph.get_edge_data(nbr, dstNode, default=0)
        edge4 = graph.get_edge_data(dstNode, nbr, default=0)
        
        if edge1 != 0:
            if edge1['weight'] == 1:
                if edge3 != 0:
                    if edge3['weight'] == 1:
                        motif_counts[0] +=1
                    else:
                        motif_counts[1] +=1
                else:
                    if edge4['weight'] == 1:
                        motif_counts[2] +=1
                    else:
                        motif_counts[3] +=1
            else:
                if edge3 != 0:
                    if edge3['weight'] == 1:
                        motif_counts[4] +=1
                    else:
                        motif_counts[5] +=1
                elif edge4 != 0:
                    if edge4['weight'] == 1:
                        motif_counts[6] +=1
                    else:
                        motif_counts[7] +=1
        else:
            if edge2['weight'] == 1:
                if edge3 != 0:
                    if edge3['weight'] == 1:
                        motif_counts[8] +=1
                    else:
                        motif_counts[9] +=1
                elif edge4 != 0:
                    if edge4['weight'] == 1:
                        motif_counts[10] +=1
                    else:
                        motif_counts[11] +=1
            else:
                if edge3 != 0:
                    if edge3['weight'] == 1:
                        motif_counts[12] +=1
                    else:
                        motif_counts[13] +=1
                elif edge4 != 0:
                    if edge4['weight'] == 1:
                        motif_counts[14] +=1
                    else:
                        motif_counts[15] +=1
    return motif_counts
                        
        
    


# Get the degree level features for one or a list of edges
# Arg:
#  node_index: dictionary, key is node id and value is index in mtx.
#  feature_store: dictionary, key is order number, valus is a list
#  files storing various configurations of mtx representing various
#  path configurations.
#  edge: tuple or np array of tuples
def getDegreeFeatures(graph, edges):
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
        
    edgeFeatures = []
    for i in range(len(src)):
        feature = degreeFeature(graph, src[i], dst[i])
        edgeFeatures.append(feature)
    print "degree level features: ", edgeFeatures
    return edgeFeatures



# Get the low order features for one or a list of edges
# Arg:
#  node_index: dictionary, key is node id and value is index in mtx.
#  feature_store: dictionary, key is order number, valus is a list
#  files storing various configurations of mtx representing various
#  path configurations.
#  edge: tuple or np array of tuples
def getLowOrderFeatures(graph, edges):
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
        
    edgeFeatures = []
    for i in range(len(src)):
        feature = lowOrderFeature(graph, src[i], dst[i])
        edgeFeatures.append(feature)
    print "Motif counts: ", edgeFeatures
    return edgeFeatures
        

 
###########for test################
if __name__ == "__main__":
    G = nx.read_weighted_edgelist("soc-sign-epinions.txt", comments='#', 
                                  create_using=nx.DiGraph(), encoding='utf-8')
    
    # print degreeFeature(G, list(G.edges(data='weight'))[1])

    # test feature extraction, write to file
    G = ex.getExampleGraph2()
    #getDegreeFeatures(G, (2, 4))
    getLowOrderFeatures(G, [ (2, 4), (5, 7) ])
    #getLowOrderFeatures(G, np.array( [ [2, 4], [5, 7] ] ))
    print '== Done =='
                                                             
    
    