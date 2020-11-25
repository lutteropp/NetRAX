import subprocess
import os

# this needs sudo apt install xvfb to run
XSERVER_MAGIC = "xvfb-run --auto-servernum --server-num=1"

DENDROSCOPE_PATH = "/home/sarah/dendroscope/Dendroscope"

NETWORK_1 = "((protopterus:0.0,(Xenopus:0.0,(((((Monodelphis:0.0,(python:0.0)#H1:0.0):0.0,(Caretta:0.0)#H2:0.0):0.0,(Homo:0.0)#H3:0.0):0.0,(Ornithorhynchus:0.0)#H4:0.0):0.0,(((#H1:0.0,((#H3:0.0,Anolis:0.0):0.0,(Gallus:0.0)#H5:0.0):0.0):0.0,(Podarcis:0.0)#H6:0.0):0.0,(((#H5:0.0,(#H6:0.0,Taeniopygia:0.0):0.0):0.0,(alligator:0.0,Caiman:0.0):0.0):0.0,(phrynops:0.0,(Emys:0.0,((Chelonoidi:0.0,#H4:0.0):0.0,#H2:0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0);"
NETWORK_2 = NETWORK_1


# Takes a string in Extended NEWICK Format, and drops the reticulation probabilities and support scores, keeping only the branch length.
def convert_newick_to_dendroscope(newick):
    new_newick = ""
    seenColon = False
    skip = False
    for c in newick:
        if c == ':':
            skip = seenColon
            seenColon = True
        elif c in [',', '(', ')', ';']:
            skip = False
            seenColon = False
        if not skip and c != '\n':
            new_newick += c
    return new_newick


# Takes two networks in Extended NEWICK format and uses Dendroscope to compute various topological distances.
def get_dendro_scores(network_1, network_2):
    cmd = "add tree=\'" + convert_newick_to_dendroscope(network_1) + convert_newick_to_dendroscope(network_2) + "\';"
    cmd += "\ncompute distance method=hardwired;"
    cmd += "\ncompute distance method=softwired;"
    cmd += "\ncompute distance method=displayedTrees;"
    cmd += "\ncompute distance method=tripartition;"
    cmd += "\ncompute distance method=nestedLabels;"
    cmd += "\ncompute distance method=pathMultiplicity;"
    cmd += "\nquit"

    temp_command_file = open("dendroscope_commands.txt", "w")
    temp_command_file.write(cmd)
    temp_command_file.close()

    dendro_cmd = XSERVER_MAGIC + " " + DENDROSCOPE_PATH + " -g -c dendroscope_commands.txt"
    print(dendro_cmd)
    dendroscope_output = subprocess.getoutput(dendro_cmd).splitlines()
    print(dendroscope_output)
    
    scores = {}
    for line in dendroscope_output:
        if line.startswith("Hardwired cluster distance:"):
            scores["hardwired_cluster_distance"] = float(line.split(": ")[1])
        elif line.startswith("Softwired cluster distance:"):
            scores["softwired_cluster_distance"] = float(line.split(": ")[1])
        elif line.startswith("Displayed trees distance:"):
            scores["displayed_trees_distance"] = float(line.split(": ")[1])
        elif line.startswith("Tripartition distance:"):
            scores["tripartition_distance"] = float(line.split(": ")[1])
        elif line.startswith("Nested labels distance:"):
            scores["nested_labels_distance"] = float(line.split(": ")[1])
        elif line.startswith("Path multiplicity distance:"):
            scores["path_multiplicity_distance"] = float(line.split(": ")[1])
    print(scores)
    os.remove("dendroscope_commands.txt")
    return scores
    
    
def evaluate(simulated_network_path, inferred_network_path):
    net1 = open(simulated_network_path).read()
    net2 = open(inferred_network_path).read()
    scores = get_dendro_scores(net1, net2)


if __name__== "__main__":
    scores = get_dendro_scores(NETWORK_1, NETWORK_2)
    print(scores)
