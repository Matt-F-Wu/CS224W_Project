# This is a helper class for extracting node2vec vectors for a DiGraph
# Note: for a node n, the feature vector for n and -n are averaged (element-wise)
# to form the final vector.

# Implementation note: the feature vector dictionary for all nodes are loaded
# into this object in memory.
import examples


class Node2vecExtractor(object):
	def __init__(self, dataset):
		super(Node2vecExtractor, self).__init__()
		self.dataset = dataset

		# read file
		feat_table = {}
		count = 0
		with open("emb/{}.emb".format(self.dataset), "r") as f:
			for row in f:
				if count == 0:
					count += 1
					continue
				row = row.strip().split()
				node, feat = abs(int(float(row[0]))), row[1:]
				for i in range(len(feat)):
					feat[i] = float(feat[i])
				if node not in feat_table:
					feat_table[node] = feat
				else:
					for i in range(len(feat_table[node])):
						feat_table[node][i] = (feat_table[node][i] + feat[i]) / 2.0
			
		self.feat_table = feat_table

	def getFeatureForEdge(self, src, dst):
		# this is essentially the dot product of the 2 embeddings.
		return [sum(map(lambda (a, b): a * b, zip(self.feat_table[int(src)], self.feat_table[int(dst)])))]


if __name__ == '__main__':
	n_extractor = Node2vecExtractor('epinions')
	# n_extractor.getFeatureForEdge(0, 1)
