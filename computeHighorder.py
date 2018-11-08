import networkx as nx
import highorder

G = nx.read_weighted_edgelist(
    "soc-sign-epinions.txt",
    comments='#',
    create_using=nx.DiGraph(),
    encoding='utf-8')

highorder.longWalkFeatureWriteAll(G, [4, 5], 'epinions')

print '== Done =='
