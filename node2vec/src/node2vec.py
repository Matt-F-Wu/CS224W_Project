import numpy as np
import networkx as nx
import random

# A performance arbitrary prob. distribution sampler using the alias method, # Reference here: http://cgi.cs.mcgill.ca/~enewel3/posts/alias-method/index.html
from categorical import Categorical

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					# I just started the walk, have no other way to go than to move 
					# deeper, namely DFS.
					walk.append(cur_nbrs[alias_nodes[cur].sample()])
				else:
					# I can do BFS, DFS, or return to previous step.
					prev = walk[-2]
					next = cur_nbrs[alias_edges[(prev, cur)].sample()]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print ('Walk iteration:')
		for walk_iter in range(num_walks):
			print (str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				# return to prev
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				# BFS
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				# DFS, go deeper
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		for i in range(len(unnormalized_probs)):
			unnormalized_probs[i] = abs(unnormalized_probs[i])
			
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return Categorical(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			
			unnormalized_probs = []
			norm_const = 0
			for nbr in sorted(G.neighbors(node)):
				abs_prob = abs(G[node][nbr]['weight'])
				unnormalized_probs.append(abs_prob)
				norm_const += abs_prob
			
			normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			
			# alias_nodes[node] contains a sampler (using the alias method) of the
			# randome walk probability distribution of all its neighbors, the probability list is sorted by the neighbor nodes.
			alias_nodes[node] = Categorical(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return
