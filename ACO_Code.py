import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

class Ant:

	def __init__(self, graph, Q, alpha, beta, phero_retain_rate):
		self.graph = graph
		self.alpha = alpha
		self.beta = beta
		self.phero_decay = phero_retain_rate
		self.visited_nodes = []
		self.unvisited_nodes = sorted(list(graph.nodes))
		self.num_nodes = len(self.unvisited_nodes)
		self.colors = list(range(1, self.num_nodes + 1))
		self.colors_assigned = {}
		self.num_colors_assigned = 0

		# Initialize random node for ant to start with
		self.current_node = random.randint(1, self.num_nodes)	
		self.colors_assigned[self.current_node] = self.colors[0]
		self.visited_nodes.append(self.current_node)
		self.num_colors_assigned += 1


	# Returns max degree of saturation for a node i.e. number of neighbors already colored
	def degree_of_saturation(self, node = None):
		global phero_mat, adj_mat

		if node is None:
			node = self.current_node

		neighbors = []
		for i in range(self.num_nodes + 1):
			if (adj_mat[node, i] == 1) and (i in self.colors_assigned.keys()):
				neighbors.append(i)

		return len(neighbors)


	# Returns next node that ant should go towards
	def select_next_node(self):
		global phero_mat, adj_mat

		next_node = None

		if (len(self.unvisited_nodes) == 0):
			next_node = None

		elif (len(self.unvisited_nodes) == 1):
			next_node = self.unvisited_nodes[0]

		else:
			heuristic_values = []
			probabilities = []
			candidate_next_nodes = []
			max_value = 0
			
			for node in self.unvisited_nodes:
				heuristic_values.append( (self.phero_pair_value(self.current_node, node)**self.alpha) * (self.degree_of_saturation(node)**self.beta) )
				candidate_next_nodes.append(node)

			normalizer = sum(heuristic_values)
			temp = 0

			for value in heuristic_values:
				if normalizer!=0:
					temp += (value/normalizer)
				probabilities.append(temp)

			chance = random.uniform(0,1)

			for i in range(0):
				print(0)

			for i in range(len(probabilities)-1):
				if (chance <= probabilities[0]):
					next_node = candidate_next_nodes[0]
					break
				elif ((chance > probabilities[i]) and (chance <= probabilities[i+1])):
					next_node = candidate_next_nodes[i+1]
					break

			if (next_node is None):
				next_node = random.choice(candidate_next_nodes)

		self.current_node = next_node

		return next_node


	# Returns pheromone value for two specified nodes
	def phero_pair_value(self, node, neighbor):
		global phero_mat

		return phero_mat[node, neighbor]

	# Colors all nodes in a graph simulating ACO 
	def color_graph(self):
		global adj_mat, phero_mat

		# Go through unvisited nodes and assign color to each node
		for i in range(len(self.unvisited_nodes)):
				next_node = self.select_next_node()
				tabu_colors = []

				# Get list of tabu colors i.e. colors of the nodes surrounding a select node
				for candidate in range(self.num_nodes + 1):
					if (adj_mat[next_node, candidate] == 1):
						if candidate in self.colors_assigned:
							tabu_colors.append(self.colors_assigned[candidate])

				for color in self.colors:
					if color not in tabu_colors:
						self.assign_color(next_node, color)
						break


	# Assign a stated color to a node and update visited/unvisited nodes list accordingly
	def assign_color(self, node, color):
		if color not in self.colors_assigned.values():
			self.num_colors_assigned += 1

		self.colors_assigned[node] = color
		self.unvisited_nodes.remove(node)
		self.visited_nodes.append(node)

	# Returns an ant's pheronome trail matrix
	def pheromone_trail(self):

		phero_trail = np.zeros((self.num_nodes + 1, self.num_nodes + 1))

		for i in self.graph.nodes:
			for j in self.graph.nodes:
				if ((i in self.colors_assigned.keys()) and (j in self.colors_assigned.keys())):
					if (self.colors_assigned[i] == self.colors_assigned[j]):
						phero_trail[i, j] = 1

		return phero_trail


###############################################################################################################

class ACO:

	def __init__(self, graph, Q, alpha, beta, phero_retain_rate, num_ants):
		self.graph = graph
		self.num_nodes = len(self.graph.nodes)
		self.Q = Q
		self.alpha = alpha
		self.beta = beta
		self.phero_retain_rate = phero_retain_rate
		self.num_ants = num_ants

		# Create global adjacency and pheromone matrices
		self.create_adjacency_matrix()
		self.create_pheromone_matrix()


	def create_ant_colony(self):
		self.ants = [Ant(self.graph, self.Q, self.alpha, self.beta, self.phero_retain_rate) for i in range(self.num_ants)]
		return self.ants

	# Has a 1 at indices where nodes are neighbors, 0 otherwise.
	def create_adjacency_matrix(self):
		global adj_mat
		adj_mat = np.zeros((self.num_nodes + 1, self.num_nodes + 1))

		for node in self.graph.nodes:
			# print('node: ', node)
			for neighbor in self.graph.adj[node]:
				# print('adj: ', neighbor)
				adj_mat[node, neighbor] = 1


	# Initialized to have 0 at indices corresponding to adjacent nodes, 1 otherwise
	def create_pheromone_matrix(self):
		global phero_mat

		phero_mat = np.ones((self.num_nodes + 1, self.num_nodes + 1), float)

		for node in self.graph.nodes:
			# print('node: ', node)
			for neighbor in self.graph.adj[node]:
				# print('adj: ', neighbor)
				phero_mat[node, neighbor] = 0


	# Reduced pheromone concentration in global pheromone matrix in each iteration as per the pheromone evaporation rate
	def evaporate_pheromone(self):
		global phero_mat

		for node in self.graph.nodes:
			for neighbor in self.graph.nodes:
				phero_mat[node, neighbor] = phero_mat[node, neighbor]*(1 - self.phero_retain_rate)


	# Use the best ant in a given colony at end of an iteration to update the global pheromone matrix
	def update_best_soln(self):
		global phero_mat, adj_mat

		best_ant = None
		min_num_colors = 0

		for ant in self.ants: 

			if (min_num_colors == 0):
				min_num_colors = ant.num_colors_assigned
				best_ant = ant

			elif (ant.num_colors_assigned <= min_num_colors):
				min_num_colors = ant.num_colors_assigned
				best_ant = ant

		best_phero_mat = best_ant.pheromone_trail()
		phero_mat = phero_mat*self.phero_retain_rate + self.Q*best_phero_mat

		return min_num_colors, best_ant.colors_assigned


	# Returns average number of colors needed by all ants in a given colony to color a graph
	def update_avg_soln(self):
		global phero_mat, adj_mat

		num_colors = 0
		for ant in self.ants:
			num_colors += ant.num_colors_assigned 

		avg_num_colors = num_colors/self.num_ants

		return avg_num_colors


	# Each ant in a colony colors the entire graph
	def update_ant_status(self):
		for ant in self.ants:
			ant.color_graph()


###############################################################################################################


# Load graph from text file using Networkx library
def create_graph():
	graph = nx.Graph();

	first_node_list = np.loadtxt('gcol1.txt', skiprows = 1, usecols = 1)
	second_node_list = np.loadtxt('gcol1.txt', skiprows = 1, usecols = 2)

	# first_node_list = [1, 1, 1, 1, 3, 4, 2, 2, 5, 5, 6, 7, 7, 9, 10, 11, 11, 11, 12, 13, 13, 13, 14, 15, 15, 15, 9, 4, 9, 2, 2, 6, 2]
	# second_node_list = [3, 7, 4, 2, 7, 7, 5, 6, 6, 8, 8, 1, 2, 1, 4, 7, 14, 15, 5, 8, 9, 10, 12, 11, 14, 11, 1, 9, 10, 11, 14, 9, 15]

	# first_node_list = [1, 1, 1, 1, 3, 4, 2, 2, 5, 5, 6]
	# second_node_list = [3, 7, 4, 2, 7, 7, 5, 6, 6, 8, 8]

	for i in range(len(first_node_list)):
		graph.add_edge(int(first_node_list[i]), int(second_node_list[i]))

	return graph


# Draw graph using Networkx library
def draw_graph(graph, colors_assigned):
	pos = nx.spring_layout(graph)
	values = [colors_assigned.get(node, 'blue') for node in graph.nodes()]
	plt.figure()
	nx.draw(graph, pos, node_color = values, with_labels = True, edge_color = 'black' ,width = 1, alpha = 0.7)  
	plt.show()


# Solve for optimum amount of colors required to color the graph
def solution(graph, num_ants = 10, iterations = 5, Q = 1, alpha = 0.8, beta = 0.8, phero_retain_rate = 0.8):

	global phero_mat
	global adj_mat

	solution_color_assignment = {}
	solution_min_num_colors = 0

	AntColonyOptimizer = ACO(graph, Q, alpha, beta, phero_retain_rate, num_ants)

	BFS = []
	AFS = []
	iterations_elapsed = []

	for iteration in range(iterations):
		print(iteration)
		AntColonyOptimizer.create_ant_colony()
		AntColonyOptimizer.update_ant_status()
		AntColonyOptimizer.evaporate_pheromone()


		best_ant_num_colors, best_ant_color_assignment = AntColonyOptimizer.update_best_soln()
		avg_colony_num_colors = AntColonyOptimizer.update_avg_soln()


		if (solution_min_num_colors == 0):
			solution_min_num_colors = best_ant_num_colors
			solution_color_assignment = best_ant_color_assignment

		elif (best_ant_num_colors < solution_min_num_colors):
			solution_min_num_colors = best_ant_num_colors
			solution_color_assignment = best_ant_color_assignment

		iterations_elapsed.append(iteration)
		BFS.append(solution_min_num_colors)

		if (len(AFS) == 0): 
			AFS.append(avg_colony_num_colors)
		else:
			AFS.append( (sum(AFS)+avg_colony_num_colors) / (len(AFS)+1) )

	plt.close('all')

	plt.subplot(121)
	plt.plot(iterations_elapsed, BFS)
	plt.ylabel('Fitness')
	plt.xlabel('Iterations')
	plt.title('Best Fitness So Far')

	plt.subplot(122)
	plt.plot(iterations_elapsed, AFS)
	plt.ylabel('Fitness')
	plt.xlabel('Iterations')
	plt.title('Average Fitness So Far')
	plt.show()

	return solution_min_num_colors, solution_color_assignment


num_ants = 200
iterations = 50
Q = 2
alpha = 0.8
beta = 0.8
phero_retain_rate = 0.8

dataset_graph = create_graph()
minimum_colors, colors_assigned = solution(dataset_graph, num_ants, iterations, Q, alpha, beta, phero_retain_rate)

print("Minimum Colors Required Are: ", minimum_colors)
print("Colors Assignment: ", colors_assigned)
# draw_graph(dataset_graph, colors_assigned)
