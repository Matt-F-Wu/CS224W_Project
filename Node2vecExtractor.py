# This is a helper class for extracting node2vec vectors for a DiGraph
# Note: for a node n, the feature vector for n and -n are averaged (element-wise)
# to form the final vector.

# Implementation note: the feature vector dictionary for all nodes are loaded
# into this object in memory.
import examples
import numpy as np

class Node2vecExtractor(object):
	def __init__(self, dataset, flag):
		super(Node2vecExtractor, self).__init__()
		
		self.dataset, self.flag = flag.split(":")

		# read file
		feat_table = {}
		count = 0
		print self.dataset
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
		# print "get ff yl"
		feat = []
		flag = self.flag.split("-")
		if "dot" in flag:
			# this is essentially the dot product of the 2 embeddings.
			
			# print "feat_table: ", len(self.feat_table[int(src)])
			# dotprod_feat = [sum(map(lambda (a, b): a * b, zip(self.feat_table[int(src)], self.feat_table[int(dst)])))]
			dotprod_feat = [np.dot(self.feat_table[int(src)],self.feat_table[int(dst)])]
			# print "hadamard: ", dotprod_feat
			feat += dotprod_feat
		
		if "hada" in flag:
			# print self.feat_table[int(src)]
			# print self.feat_table[int(dst)]
			hada_feat = np.prod([self.feat_table[int(src)],self.feat_table[int(dst)]], axis=0)
			hada_feat = list(hada_feat)
			# print hada_feat
			# print "hada_feat: ", len(hada_feat)
			feat += hada_feat

		if "concat" in flag:
			concat_feat = self.feat_table[int(src)] + self.feat_table[int(dst)]
			# print "concat: ", len(concat_feat)
			feat += concat_feat
		
		if "sum" in flag:
			sum_feat = np.array(self.feat_table[int(src)]) + np.array(self.feat_table[int(dst)])
			sum_feat = list(sum_feat)
			# print "sum: ", len(sum_feat)
			feat += sum_feat
		
		if "avg" in flag:
			avg_feat = np.array(self.feat_table[int(src)]) + np.array(self.feat_table[int(dst)])/2.0
			avg_feat = list(avg_feat)
			# print "avg: ", len(avg_feat)
			feat += avg_feat
		
		if "dis" in flag:
			dis_feat = [np.linalg.norm(np.array(self.feat_table[int(src)]) - np.array(self.feat_table[int(dst)]))]
			# print "dis: ", dis_feat
			feat += dis_feat
		# print "node2vec feat: ", feat
		return feat


if __name__ == '__main__':
	# print "here yl"
	n_extractor = Node2vecExtractor('wiki_lowq',"hada")
	n_extractor.getFeatureForEdge(2, 1)
	# n_extractor.getFeatureForEdge(0, 1, "concat")
	# n_extractor.getFeatureForEdge(0, 1, "sum")
	# n_extractor.getFeatureForEdge(0, 1, "avg")
	# n_extractor.getFeatureForEdge(0, 1, "distance")
	# print "finish"
