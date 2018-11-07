#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:56:03 2018

@author: liguo
"""
import networkx as nx
from random import choice

import highorder


# Degree type feature and lowOrder feature:
def degreeFeature(graph, edge): # TUNGraphEdgeI
    # get source and destination node id for given edge in given graph
    srcNode, dstNode, weight= edge
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


 
###########for test################
if __name__ == "__main__":
    G = nx.read_weighted_edgelist("soc-sign-epinions.txt", comments='#', 
                                  create_using=nx.DiGraph(), encoding='utf-8')
    
    # print degreeFeature(G, list(G.edges(data='weight'))[1])

    # test high order feature extraction, write to file
    highorder.longWalkFeatureWriteAll(G, [4, 5], 'epinions')

    print '== Done =='
                                                             
    
    