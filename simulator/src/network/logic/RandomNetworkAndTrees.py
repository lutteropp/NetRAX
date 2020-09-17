import numpy as np
import random
import networkx as nx
import sys
import copy
import collections
import subprocess
import matplotlib


class SimulationParameters:
    def __init__(self):
        self.time_limit = 0.1  # height of the network
        self.speciation_rate = 20.0  # speciation rate
        self.hybridization_rate = 10.0  # hybridization rate
        self.pop_size = 0.01  # pop size, not used in our non_ILS simulations
        # to print inheritance probs (Dendroscope cannot read them, so we print also a network without them
        self.inheritance = True
        self.ILS = True  # non_ILS simulations for now
        self.number_trees = 1  # number of different trees to generate
        self.number_sites = 1  # number of sites per tree
        self.output = "test"  # basename for the output files


############### CONVERT TO NEWICK ##############
def Newick_From_MULTree(params, tree, root, hybrid_nodes):
    if tree.out_degree(root) == 0:
        if root in hybrid_nodes:
            return "#H"+str(hybrid_nodes[root])
        return str(root)
    Newick = ""
    for v in tree.successors(root):
        if not(params.ILS):
            Newick += Newick_From_MULTree(params, tree, v,
                                          hybrid_nodes)+":"+str(tree[root][v]['length'])
        else:
            coalescent_time = tree[root][v]['length']/params.pop_size
            Newick += Newick_From_MULTree(params, tree,
                                          v, hybrid_nodes)+":"+str(coalescent_time)
        if params.inheritance:
            if v in hybrid_nodes:
                Newick += "::"+str(tree[root][v]['prob'])
        Newick += ","
    Newick = "("+Newick[:-1]+")"
    if root in hybrid_nodes:
        Newick += "#H"+str(hybrid_nodes[root])
    return Newick


def draw_network(params, nw):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout
    plt.title('draw_networkx')
    pos = graphviz_layout(nw, prog='dot')
    nx.draw(nw, pos, with_labels=False, arrows=True)
    plt.savefig(params.output+'_graph.png')


def simulate_network(params):
    file = open(params.output+"_trees", "w")
    fileNetwork = open(params.output+"_network", "w")
    fileNetworkDendroscope = open(params.output+"_networkDendroscope", "w")

    simulateNetwork = 1

    while simulateNetwork == 1:
        simulateNetwork = 0
        nw = nx.DiGraph()
        nw.add_node(0)
        leaves = set([0])
        current_node = 1

        extra_time = np.random.exponential(1/float(params.speciation_rate))
        current_time = extra_time
        current_speciation_rate = float(params.speciation_rate)
        current_hybridization_rate = float(0)
        rate = current_speciation_rate + current_hybridization_rate

        # First create a MUL-tree
        hybrid_nodes = dict()
        no_of_hybrids = 0

        while current_time < params.time_limit:
            if random.random() < current_speciation_rate / rate:
                # speciate
                splitting_leaf = random.choice(list(leaves))
                nw.add_weighted_edges_from(
                    [(splitting_leaf, current_node, 0), (splitting_leaf, current_node+1, 0)], weight='length')
                leaves.remove(splitting_leaf)
                leaves.add(current_node)
                leaves.add(current_node+1)
                current_node += 2
            else:
                # Hybridize
                no_of_hybrids += 1
                merging = random.sample(leaves, 2)
                l0 = merging[0]
                l1 = merging[1]
                pl0 = -1
                for p in nw.predecessors(l0):
                    pl0 = p
                pl1 = -1
                for p in nw.predecessors(l1):
                    pl1 = p
                nw.add_weighted_edges_from(
                    [(l0, current_node, 0)], weight='length')
                leaves.remove(l0)
                leaves.remove(l1)
                leaves.add(current_node)
                prob = random.random()
                nw[pl0][l0]['prob'] = prob
                nw[pl1][l1]['prob'] = 1-prob
                hybrid_nodes[l0] = no_of_hybrids
                hybrid_nodes[l1] = no_of_hybrids
                current_node += 1
            # Now extend all pendant edges
            for l in leaves:
                pl = -1
                for p in nw.predecessors(l):
                    pl = p
                nw[pl][l]['length'] += extra_time
            no_of_leaves = len(leaves)
            current_speciation_rate = float(
                params.speciation_rate*no_of_leaves)
            current_hybridization_rate = float((no_of_leaves * (no_of_leaves - 1))/2*params.hybridization_rate)
            rate = current_speciation_rate + current_hybridization_rate
            extra_time = np.random.exponential(1/rate)
            current_time += extra_time

        extra_time -= current_time-params.time_limit

        if(len(leaves) > 3):  # network contains more than 3 sequences
            for l in leaves:
                pl = -1
                for p in nw.predecessors(l):
                    pl = p
                nw[pl][l]['length'] += extra_time

            hybrid_nodes_fake = dict()

            if not(params.ILS):
                for t in range(0, params.number_trees):
                    # generate a subdivision
                    tree = copy.deepcopy(nw)

                    m = int(len(hybrid_nodes)/2)

                    parents = np.zeros((m+1, 2))

                    for k in hybrid_nodes.keys():
                        if (parents[hybrid_nodes.get(k)][0] == 0):
                            parents[hybrid_nodes.get(k)][0] = k
                        else:
                            parents[hybrid_nodes.get(k)][1] = k

                    for h in range(1, m+1):
                        p = random.random()
                        father1 = -1
                        father2 = -1
                        node1 = int(parents[h][0])
                        node2 = int(parents[h][1])
                        for f in tree.predecessors(node1):
                            father1 = f
                        for f in tree.predecessors(node2):
                            father2 = f

                        toDelete = -1  # parent to delete
                        fatherNodeToDelete = -1  # parent of the parent to delete
                        toKeep = -1  # parent to keep
                        fatherNodeToKeep = -1  # parent of the parent of the parent to delete

                        if (p < tree[father1][node1]['prob']):
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
                        else:
                            tree.add_weighted_edges_from(
                                [(fatherNodeToKeep, toDelete, tree[fatherNodeToKeep][toKeep]['length'])], weight='length')
                            tree.remove_edge(fatherNodeToDelete, toDelete)
                            tree.remove_edge(fatherNodeToKeep, toKeep)
                            tree.remove_node(toKeep)  # yep, poorly chosen name

                    # clean the subdivision to get an actual phylogenetic tree (no fake leaves/roots and
                    # no  in- and out-degree 1 nodes
                    changed = 1

                    while(changed == 1):
                        changed = 0
                        nodes = tree.nodes()
                        for v in nodes:
                            if (tree.out_degree(v) == 0) and not(v in leaves) and tree.in_degree(v) > 0:
                                # remove a fake leaf
                                father = -1
                                for f in tree.predecessors(v):
                                    father = f
                                tree.remove_edge(father, v)
                                tree.remove_node(v)
                                changed = 1
                                break
                            elif(tree.out_degree(v) == 0) and not(v in leaves) and tree.in_degree(v) == 0:
                                # remove a fake root
                                tree.remove_node(v)
                                changed = 1
                                break
                            elif (tree.out_degree(v) == 1) and (tree.in_degree(v) == 1):
                                # remove a node with in- and out-degree 1
                                father = -1
                                child = -1
                                for f in tree.predecessors(v):
                                    father = f
                                for c in tree.successors(v):
                                    child = c
                                lengthNewEdge = tree[father][v]['length'] + \
                                    tree[v][child]['length']
                                tree.add_weighted_edges_from(
                                    [(father, child, lengthNewEdge)], weight='length')
                                tree.remove_edge(father, v)
                                tree.remove_edge(v, child)
                                tree.remove_node(v)
                                changed = 1
                                break

                    file.write("["+str(params.number_sites)+"]" +
                               Newick_From_MULTree(params, tree, 0, hybrid_nodes_fake)+";\n")

            fileNetwork.write(Newick_From_MULTree(
                params, nw, 0, hybrid_nodes)+";\n")
            params.inheritance = False
            fileNetworkDendroscope.write(
                Newick_From_MULTree(params, nw, 0, hybrid_nodes)+";\n")
            print('done')
        else:
            print('The simulated network contains less than 4 leaves, try again')
            simulateNetwork = 1
        
        #draw_network(params, nw)
    file.close()
    fileNetwork.close()
    fileNetworkDendroscope.close()
    return no_of_leaves, no_of_hybrids


def simulate_network_and_sequences(params):
    n_taxa, n_reticulations = simulate_network(params)
    total_length = params.number_trees * params.number_sites
    cmd = 'seq-gen -mHKY -t3.0 -f0.3,0.2,0.2,0.3 -l'+str(total_length)+'-p'+str(params.number_trees)+' < '+params.output+'_trees > '+params.output+'.dat'
    subprocess.getoutput(cmd)
    return n_taxa, n_reticulations


def parse_user_input():
    params = SimulationParameters()
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-t":
            i += 1
            params.time_limit = float(sys.argv[i])
        if arg == "-sp":
            i += 1
            params.speciation_rate = float(sys.argv[i])
        if arg == "-hyb":
            i += 1
            params.hybridization_rate = float(sys.argv[i])
        if arg == "-no_inheritance":
            i += 1
            params.inheritance = False
        if arg == "-no_ILS":
            i += 1
            params.ILS = False
        if arg == "-o":
            i += 1
            params.output = sys.argv[i]
        if arg == "-nb_trees":
            i += 1
            params.number_trees = int(sys.argv[i])
        if arg == "-nb_sites":
            i += 1
            params.number_sites = int(sys.argv[i])
        if arg == "-pop_size":
            i += 1
            params.pop_size = float(sys.argv[i])
        i += 1
    return params


if __name__ == "__main__":
    params = parse_user_input()
    simulate_network(params)
