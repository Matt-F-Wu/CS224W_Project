import numpy as np
import networkx as nx
import random

# Hao: enable multi-threading for performance speed up.
from multiprocessing.dummy import Pool as ThreadPool
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

		sign = 1
		while len(walk) < walk_length:
			cur = abs(walk[-1])
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					# I just started the walk, have no other way to go than to move 
					# deeper, namely DFS.
					next = cur_nbrs[alias_nodes[cur].sample()]
					sign = sign * G[cur][next]['weight']
					walk.append(sign * next)
				else:
					# I can do BFS, DFS, or return to previous step.
					prev = abs(walk[-2])
					next = cur_nbrs[alias_edges[(prev, cur)].sample()]
					sign = sign * G[cur][next]['weight']
					walk.append(sign * next)
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
		pool = ThreadPool(8) 

		print ('Walk iteration:')
		for walk_iter in range(num_walks):
			print (str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			# Each walk is independently parallalizable, thus I will distribute
			# the workload to 4 threads to speed up performance.
			pool.map(
					lambda node: walks.append(self.node2vec_walk(
							walk_length=walk_length, start_node=node)), nodes)

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

		return normalized_probs and Categorical(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}

		def assign_probablistic_sampler_per_node(node):
			unnormalized_probs = []
			norm_const = 0
			for nbr in sorted(G.neighbors(node)):
				abs_prob = abs(G[node][nbr]['weight'])
				unnormalized_probs.append(abs_prob)
				norm_const += abs_prob
			
			normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			
			# alias_nodes[node] contains a sampler (using the alias method) of the
			# randome walk probability distribution of all its neighbors, the probability list is sorted by the neighbor nodes.
			alias_nodes[node] = normalized_probs and Categorical(normalized_probs)

		# spin up 4 threads
		map(assign_probablistic_sampler_per_node, G.nodes())

		alias_edges = {}
		triads = {}

		if is_directed:
			def assign_sampler_per_edge_dir(edge):
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
			
			map(assign_sampler_per_edge_dir, G.edges())
		else:
			def assign_sampler_per_edge(edge):
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
			map(assign_sampler_per_edge, G.edges())

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return
