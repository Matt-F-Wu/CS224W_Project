import sys
import getopt
import networkx as nx
import highorder

optval, leftover = getopt.getopt(sys.argv[1:], 'd:')

dataset = 'epinions'

for o, v in optval:
  if o == '-d':
    dataset = v

filenameMap = {
  'epinions': "soc-sign-epinions.txt",
  'wiki': 'wiki.txt',
  'slashdot': 'soc-sign-Slashdot081106.txt'
}

G = nx.read_weighted_edgelist(
    filenameMap[dataset],
    comments='#',
    create_using=nx.DiGraph(),
    encoding='utf-8')

highorder.longWalkFeatureWriteAll(G, [4, 5], dataset)

print '== Done =='
