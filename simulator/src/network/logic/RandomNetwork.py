import numpy as np
import random
import networkx as nx
import sys




###############################2. I/O############################

i = 1
while i < len(sys.argv):
    arg= sys.argv[i]
    if arg == "-t":
        i+=1
        time_limit = float(sys.argv[i])
    if arg == "-sp":
        i+=1
        speciation_rate = float(sys.argv[i])
    if arg == "-hyb":
        i+=1
        hybridization_rate = float(sys.argv[i])
    if arg == "-no_inheritence":
        inheritence = False
    i += 1




simulateNetwork = True
    
while simulateNetwork:
	simulateNetwork = False

	##PARAMETERS
	#time_limit = np.random.exponential(0.2) 
	d = np.random.exponential(10)
	r = random.random()
	#if r == 0:
	#    speciation_rate = d
	#else:
	#    speciation_rate = float(d/(1-r))
	#hybridization_rate = float(r * speciation_rate ) 

	#inheritence = True
	time_limit = np.random.exponential(0.2) + 0.1 
	speciation_rate = random.random()*20 + 5
	hybridization_rate = float(speciation_rate * 0.003)
	##

	print(str(speciation_rate))
	
	nw = nx.DiGraph()
	nw.add_node(0)
	leaves = set([0])
	current_node = 1

	extra_time = np.random.exponential(1/float(speciation_rate))
	current_time = extra_time
	current_speciation_rate    = float(speciation_rate)
	current_hybridization_rate = float(0)
	rate = current_speciation_rate + current_hybridization_rate

	#First create a MUL-tree
	hybrid_nodes=dict()
	no_of_hybrids = 0

	while current_time<time_limit:
		if random.random() < current_speciation_rate / rate:
			#speciate
			splitting_leaf = random.choice(list(leaves))
			nw.add_weighted_edges_from([(splitting_leaf,current_node,0),(splitting_leaf,current_node+1,0)], weight = 'length')
			leaves.remove(splitting_leaf)
			leaves.add(current_node)
			leaves.add(current_node+1)
			current_node+=2
		else:
			#Hybridize
			no_of_hybrids+=1
			merging = random.sample(leaves,2)
			l0 = merging[0]
			l1 = merging[1]
			pl0 = -1
			for p in nw.predecessors(l0):
				pl0=p
			pl1 = -1
			for p in nw.predecessors(l1):
				pl1=p
			nw.add_weighted_edges_from([(l0,current_node,0)],weight = 'length')
			leaves.remove(l0)
			leaves.remove(l1)
			leaves.add(current_node)
			prob = random.random()
			nw[pl0][l0]['prob'] = prob
			nw[pl1][l1]['prob'] = 1-prob
			hybrid_nodes[l0]=no_of_hybrids
			hybrid_nodes[l1]=no_of_hybrids
			current_node+=1 
		#Now extend all pendant edges
		for l in leaves:
			pl = -1
			for p in nw.predecessors(l):
				pl = p
			nw[pl][l]['length']+=extra_time
		no_of_leaves = len(leaves)
		current_speciation_rate    = float(speciation_rate*no_of_leaves)
		current_hybridization_rate = float(hybridization_rate*(no_of_leaves * (no_of_leaves - 1))/2)
		rate = current_speciation_rate + current_hybridization_rate
		extra_time    = np.random.exponential(1/rate)
		current_time += extra_time

	if current_node == 1: ## add this check to avoid the simulator to complain 
		#print("Only one node in the network;") 
		print("to="+str(time_limit)+",d="+str(d)+",r="+str(r)+",lambda="+str(speciation_rate)+",nu="+str(hybridization_rate)+",no_of_leaves="+str(len(leaves))+",no_of_hybrids="+str(no_of_hybrids)+ ",ratio="+ str(float(no_of_hybrids/len(leaves))))
	else:

		extra_time -= current_time-time_limit
		for l in leaves:
			pl = -1
			for p in nw.predecessors(l):
				pl = p
			nw[pl][l]['length']+=extra_time


	############### NOW CONVERT TO NEWICK ##############

		def Newick_From_MULTree(tree,root,hybrid_nodes):
			if tree.out_degree(root)==0:
				if root in hybrid_nodes:
					return "#H"+str(hybrid_nodes[root])
				return str(root)
			Newick = ""
			for v in tree.successors(root):
				Newick+= Newick_From_MULTree(tree,v,hybrid_nodes)+":"+str(tree[root][v]['length'])
				if inheritence:
					if v in hybrid_nodes:
						Newick+="::"+str(tree[root][v]['prob'])
				Newick+= ","
			Newick = "("+Newick[:-1]+")"
			if root in hybrid_nodes:
				Newick += "#H"+str(hybrid_nodes[root])
			return Newick
		print("to="+str(time_limit)+",d="+str(d)+",r="+str(r)+",lambda="+str(speciation_rate)+",nu="+str(hybridization_rate)+",no_of_leaves="+str(len(leaves))+",no_of_hybrids="+str(no_of_hybrids)+ ",ratio="+ str(float(no_of_hybrids/len(leaves))))
		#print(Newick_From_MULTree(nw,0,hybrid_nodes)+";")
		#inheritence = False
		#print(Newick_From_MULTree(nw,0,hybrid_nodes)+";")

	if ( len(leaves) > 30 and float(no_of_hybrids/len(leaves)) < 0.1):
	#if ( len(leaves) < 100 and no_of_hybrids < float(len(leaves)/3)):  ## add this check to avoid the simulator to complain 
		print("OK")
	else : 
		print("trying again")
		simulateNetwork = True
		
		
