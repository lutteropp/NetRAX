import numpy as np
import random
import networkx as nx
import sys
import copy 
import collections

##PARAMETERS
time_limit = 0.1 #height of the network
speciation_rate = 20.0 #speciation rate
hybridization_rate = 10.0 # hybridization rate
pop_size=0.01 # pop size, not used in our non_ILS simulations
inheritance = True # to print inheritance probs (Dendroscope cannot read them, so we print also a network without them
ILS = True # non_ILS simulations for now
number_trees=1 #number of different trees to generate
output="test" #basename for the output files
##

############### CONVERT TO NEWICK ##############

def Newick_From_MULTree(tree,root,hybrid_nodes):
    if tree.out_degree(root)==0:
        if root in hybrid_nodes:
            return "#H"+str(hybrid_nodes[root])
        return str(root)
    Newick = ""
    for v in tree.successors(root):
    	if not(ILS):
        	Newick+= Newick_From_MULTree(tree,v,hybrid_nodes)+":"+str(tree[root][v]['length'])
    	else:
        	coalescent_time = tree[root][v]['length']/pop_size
        	Newick+= Newick_From_MULTree(tree,v,hybrid_nodes)+":"+str(coalescent_time)
    	if inheritance:
            if v in hybrid_nodes:
                Newick+="::"+str(tree[root][v]['prob'])
    	Newick+= ","
    Newick = "("+Newick[:-1]+")"
    if root in hybrid_nodes:
        Newick += "#H"+str(hybrid_nodes[root])
    return Newick



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
    if arg == "-no_inheritance":
    	i += 1
    	inheritance = False
    if arg == "-no_ILS":
    	i += 1
    	ILS = False
    if arg == "-nb_tree":
    	i += 1
    	number_trees = int(sys.argv[i])
    if arg == "-o":
    	i += 1
    	output = sys.argv[i]
    if arg == "-pop_size":
    	i += 1
    	pop_size = float(sys.argv[i])
    i += 1

file = open(output+"_trees","w") 
fileNetwork = open(output+"_network","w") 

simulateNetwork=1

while simulateNetwork==1:
 	simulateNetwork=0
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
 	 	current_hybridization_rate = float((no_of_leaves * (no_of_leaves - 1))/2)
 	 	rate = current_speciation_rate + current_hybridization_rate
 	 	extra_time    = np.random.exponential(1/rate)
 	 	current_time += extra_time


 	extra_time -= current_time-time_limit

 	if(len(leaves)>2):#network contains less than 2 sequences
 	 	for l in leaves:
 	 		pl = -1
 	 		for p in nw.predecessors(l):
 	 	 	 	pl = p
 	 		nw[pl][l]['length']+=extra_time

 	 	hybrid_nodes_fake=dict()

 	 	if not(ILS):
 	 		for t in range(0,number_trees):
 	 	 	 	#generate a subdivision 
 	 	 	 	tree = copy.deepcopy(nw)

 	 	 	 	m=int(len(hybrid_nodes)/2);

 	 	 	 	parents= np.zeros((m+1,2))
 	 	
 	 	 	 	for k in hybrid_nodes.keys():
 	 	 	 		if (parents[hybrid_nodes.get(k)][0]==0):
 	 	 	 	 	 	parents[hybrid_nodes.get(k)][0]=k
 	 	 	 		else:
 	 	 	 	 	 	parents[hybrid_nodes.get(k)][1]=k	
 	 	 	 	
 	 	 	 	for h in range(1, m+1):
 	 	 	 		p=random.random()
 	 	 	 		father1 = -1
 	 	 	 		father2 = -1
 	 	 	 		node1 = int(parents[h][0])
 	 	 	 		node2 = int(parents[h][1])
 	 	 	 		for f in tree.predecessors(node1):
 	 	 	 	 	 	father1 = f
 	 	 	 		for f in tree.predecessors(node2):
 	 	 	 	 	 	father2 = f
 	 		
 	 	 	 		toDelete = -1 #parent to delete
 	 	 	 		fatherNodeToDelete = -1  #parent of the parent to delete
 	 	 	 		toKeep = -1 #parent to keep
 	 	 	 		fatherNodeToKeep = -1  #parent of the parent of the parent to delete
	
 	 	 	 		if ( p<tree[father1][node1]['prob'] ):
 	 	 	 	 	 	toDelete = node1 
 	 	 	 	 	 	toKeep = node2 
 	 	 	 	 	 	fatherNodeToDelete = father1
 	 	 	 	 	 	fatherNodeToKeep = father2

 	 	 	 		else:
 	 	 	 	 	 	toDelete = node2 
 	 	 	 	 	 	toKeep = node1
 	 	 	 	 	 	fatherNodeToDelete = father2
 	 	 	 	 	 	fatherNodeToKeep = father1
 	 	
 	 	 	 		if tree.out_degree(toDelete) == 0 and tree.in_degree(toDelete) > 0:
 	 	 	 	 	 	tree.remove_edge(fatherNodeToDelete, toDelete)	
 	 	 	 	 	 	tree.remove_node(toDelete)

 	 	 	 		elif tree.out_degree(toDelete) == 0 and tree.in_degree(toDelete) == 0:
 	 	 	 	 	 	tree.remove_node(toDelete)
 	 	 	 		else :
 	 	 	 	 	 	tree.add_weighted_edges_from([(fatherNodeToKeep,toDelete,tree[fatherNodeToKeep][toKeep]['length'])],weight = 'length')
 	 	 	 	 	 	tree.remove_edge(fatherNodeToDelete, toDelete)	
 	 	 	 	 	 	tree.remove_edge(fatherNodeToKeep, toKeep)	
 	 	 	 	 	 	tree.remove_node(toKeep) #yep, poorly chosen name
 	 		
 	 	 	 	#clean the subdivision to get an actual phylogenetic tree (no fake leaves/roots and
 	 	 	 	# no  in- and out-degree 1 nodes
 	 	 	 	changed=1
 	 	 	 	 
 	 	 	 	while(changed==1):
 	 	 	 		changed=0
 	 	 	 		nodes=tree.nodes()
 	 	 	 		for v in nodes:
 	 	 	 	 	 	if (tree.out_degree(v) == 0) and not(v in leaves) and tree.in_degree(v) > 0:
 	 	 	 	 	 		#remove a fake leaf
 	 	 	 	 	 		father=-1
 	 	 	 	 	 		for f in tree.predecessors(v):
 	 	 	 	 	 	 	 	father = f
 	 	 	 	 	 		tree.remove_edge(father,v)	
 	 	 	 	 	 		tree.remove_node(v)
 	 	 	 	 	 		changed=1
 	 	 	 	 	 		break
 	 	 	 	 	 	elif(tree.out_degree(v) == 0) and not(v in leaves) and tree.in_degree(v) == 0:
 	 	 	 	 	 		#remove a fake root
 	 	 	 	 	 		tree.remove_node(v)
 	 	 	 	 	 		changed=1
 	 	 	 	 	 		break
 	 	 	 	 	 	elif (tree.out_degree(v) == 1) and (tree.in_degree(v) == 1):
 	 	 	 	 	 		#remove a node with in- and out-degree 1
 	 	 	 	 	 		father=-1
 	 	 	 	 	 		child=-1
 	 	 	 	 	 		for f in tree.predecessors(v):
 	 	 	 	 	 	 	 	father = f
 	 	 	 	 	 		for c in tree.successors(v):
 	 	 	 	 	 	 	 	child = c
 	 	 	 	 	 		lengthNewEdge = tree[father][v]['length'] + tree[v][child]['length']	
 	 	 	 	 	 		tree.add_weighted_edges_from([(father,child,lengthNewEdge)],weight = 'length')
 	 	 	 	 	 		tree.remove_edge(father,v)
 	 	 	 	 	 		tree.remove_edge(v,child)
 	 	 	 	 	 		tree.remove_node(v)
 	 	 	 	 	 		changed=1 
 	 	 	 	 	 		break
 	 		
 	 	 	 	file.write("[1]"+Newick_From_MULTree(tree,0,hybrid_nodes_fake)+";\n")

 	 	fileNetwork.write(Newick_From_MULTree(nw,0,hybrid_nodes)+";\n")	
 	 	inheritance = False
 	 	fileNetwork.write(Newick_From_MULTree(nw,0,hybrid_nodes)+";\n")	
 	 	print('done')
 	else:
 	 	print('The simulated newtork contains less than 3 leaves, try again')	
 	 	simulateNetwork=1
	