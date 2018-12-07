#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:03:45 2018

@author: liguo
"""

import snap
import numpy as np
import matplotlib.pyplot as plt
import random
import csv


def degreeDistribution(G):
    print "Nodes number: ", G.GetNodes()  # 37444
    print "Edges number: ", G.GetEdges()  # 561119
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(G, DegToCntV)
    X = []
    Y = []
    for item in DegToCntV:
        X.append(item.GetVal1())
        Y.append(item.GetVal2())
    plt.loglog(X, Y, linestyle = 'dotted', color = 'b', label = 'Collaboration Network')
    plt.ylabel('Number of Users')
    plt.xlabel('Number of Neighbors (degree)')
    plt.show()
    
def clustCoefDistribution(G):

    print "Nodes number: ", G.GetNodes()  # 37444
    print "Edges number: ", G.GetEdges()  # 561119
    DegToCCfV = snap.TFltPrV()
    snap.GetClustCfAll(G, DegToCCfV)
    X = []
    Y = []
    for item in DegToCCfV:
        X.append(item.GetVal1())  # degree
        Y.append(item.GetVal2())  # avg. clustering coefficient
    plt.plot(X, Y, 'ro', label = 'Collaboration Network')
    plt.ylabel('Average Clustering Coefficient')
    plt.xlabel('Number of Neighbors (degree)')
    plt.show()
    
    

    
if __name__ == '__main__':
    G = snap.LoadEdgeList(snap.PNGraph, "wiki.txt", 0, 1)
    degreeDistribution(G)
    clustCoefDistribution(G)