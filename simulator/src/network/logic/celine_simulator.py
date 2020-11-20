import numpy as np
import random
import networkx as nx
import sys


############################### I/O############################

class CelineParams:
    def __init__(self):
        self.time_limit = np.random.exponential(0.2) + 0.1
        self.speciation_rate = random.random()*20 + 5
        self.hybridization_rate = float(self.speciation_rate * 0.003)
        self.inheritance = True
        self.wanted_taxa = -1
        self.wanted_reticulations = -1
        self.min_taxa = -1
        self.max_taxa = -1
        self.min_reticulations = -1
        self.max_reticulations = -1


def parse_user_input():
    params = CelineParams()
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
            params.inheritance = False
        if arg == "-ntaxa":
            i += 1
            params.wanted_taxa = int(sys.argv[i])
        if arg == "-nreticulations":
            i += 1
            params.wanted_reticulations = int(sys.argv[i])
        if arg == "-min_taxa":
            i += 1
            params.min_taxa = int(sys.argv[1])
        if arg == "-max_taxa":
            i += 1
            params.max_taxa = int(sys.argv[1])
        if arg == "-min_reticulations":
            i += 1
            params.min_reticulations = int(sys.argv[1])
        if arg == "-max_reticulations":
            i += 1
            params.max_reticulations = int(sys.argv[1])
        i += 1
    return params


############### CONVERT TO NEWICK ##############
def Newick_From_MULTree(tree, root, hybrid_nodes, params):
    if tree.out_degree(root) == 0:
        if root in hybrid_nodes:
            return "#H"+str(hybrid_nodes[root])
        return str(root)
    Newick = ""
    for v in tree.successors(root):
        Newick += Newick_From_MULTree(tree, v, hybrid_nodes, params) + \
                                      ":"+str(tree[root][v]['length'])
        if params.inheritance:
            if v in hybrid_nodes:
                Newick += "::"+str(tree[root][v]['prob'])
        Newick += ","
    Newick = "("+Newick[:-1]+")"
    if root in hybrid_nodes:
        Newick += "#H"+str(hybrid_nodes[root])
    return Newick


def reshuffle_params(params):
    params.time_limit = np.random.exponential(0.2) + 0.1
    params.speciation_rate = random.random()*20 + 5
    params.hybridization_rate = float(params.speciation_rate * 0.003)
    return params


def check_counts(n_taxa, n_reticulations, params):
    taxa_ok = False
    reticulations_ok = False
    # check number of taxa
    if params.wanted_taxa != -1:
        taxa_ok = (n_taxa == params.wanted_taxa)
    elif params.min_taxa != -1 and params.max_taxa != -1:
        taxa_ok = (n_taxa >= params.min_taxa and n_taxa <= params.max_taxa)
    else:
        taxa_ok = (n_taxa >= 30)
    # check number of reticulations
    if params.wanted_reticulations != -1:
        reticulations_ok = (n_reticulations == params.wanted_reticulations)
    elif params.min_reticulations != -1 and params.max_reticulations != -1:
        reticulations_ok = (n_reticulations >= params.min_reticulations and n_reticulations <= params.max_reticulations)
    else:
        reticulations_ok = (float(n_reticulations)/n_taxa <= 0.1)
    
    return (taxa_ok and reticulations_ok)


def simulate_network_step(params):
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
            nw.add_weighted_edges_from([(l0, current_node, 0)], weight='length')
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
        current_speciation_rate = float(params.speciation_rate*no_of_leaves)
        current_hybridization_rate = float(
            params.hybridization_rate*(no_of_leaves * (no_of_leaves - 1))/2)
        rate = current_speciation_rate + current_hybridization_rate
        extra_time = np.random.exponential(1/rate)
        current_time += extra_time

    if current_node != 1:  # add this check to avoid the simulator to complain about Only one node in the network
        extra_time -= current_time-params.time_limit
        for l in leaves:
            pl = -1
            for p in nw.predecessors(l):
                pl = p
            nw[pl][l]['length'] += extra_time
        
    print("to="+str(params.time_limit)+",lambda="+str(params.speciation_rate)+",nu="+str(params.hybridization_rate) +
          ",no_of_leaves="+str(len(leaves))+",no_of_hybrids="+str(no_of_hybrids) + ",ratio=" + str(float(no_of_hybrids/len(leaves))))
       
    # if ( len(leaves) < 100 and no_of_hybrids < float(len(leaves)/3)):  ## add this check to avoid the simulator to complain

    if check_counts(len(leaves), no_of_hybrids, params):
        n_taxa = len(leaves)
        n_reticulations = no_of_hybrids
        newick = Newick_From_MULTree(nw,0,hybrid_nodes,params)+";"
        param_info = {}
        param_info["to"] = params.time_limit
        param_info["lambda"] = params.speciation_rate
        param_info["nu"] = params.hybridization_rate
        param_info["no_of_leaves"] = len(leaves)
        param_info["no_of_hybrids"] = no_of_hybrids
        param_info["ratio"] = float(no_of_hybrids/len(leaves))
        return (n_taxa, n_reticulations, newick, param_info)
    else: 
        return None



def simulate_network(params):
    res = simulate_network_step(params)
    while res == None:
        print("trying again")
        params = reshuffle_params(params)
        res = simulate_network_step(params)
    print("OK")
    return res    
    
        
def simulate_network_celine(wanted_taxa, wanted_reticulations, network_path):
    params = CelineParams()
    params.wanted_taxa = wanted_taxa
    params.wanted_reticulations = wanted_reticulations
    n_taxa, n_reticulations, newick, param_info = simulate_network(params)
    network_file = open(network_path, "w")
    network_file.write(newick)
    network_file.close()
    return param_info
    

def simulate_network_celine_minmax(min_taxa, max_taxa, min_reticulations, max_reticulations, network_path):
    params = CelineParams()
    params.min_taxa = min_taxa
    params.max_taxa = max_taxa
    params.min_reticulations = min_reticulations
    params.max_reticulations = max_reticulations
    n_taxa, n_reticulations, newick, param_info = simulate_network(params)
    network_file = open(network_path, "w")
    network_file.write(newick)
    network_file.close()
    return param_info
        
        
if __name__ == "__main__":
    params = parse_user_input()
    n_taxa, n_reticulations, newick, param_info = simulate_network(params)
    print(newick)
