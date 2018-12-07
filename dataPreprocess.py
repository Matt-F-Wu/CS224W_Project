#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:38:17 2018

@author: liguo
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
    
def signDistribution(G):
    neg = 0
    pos = 0
    for edge in G.edges():
        srcNode, dstNode = edge
        weight = G[srcNode][dstNode]['weight']
        if weight == -1:
            neg += 1
        else:
            pos += 1
    Sum = neg + pos + 0.0
    x = np.array([-1, 1])
    plt.bar(x, [neg/Sum, pos/Sum])
    plt.xticks(x, ['-1','1'])
    plt.xlabel("Sign of Edges in Wiki Election dataset")
    plt.ylabel("Probability of Users")
    
    
    
    
if __name__ == '__main__':
    G = nx.read_weighted_edgelist("wiki.txt", comments='#', 
                                  create_using=nx.DiGraph(), encoding='utf-8')
    signDistribution(G)

