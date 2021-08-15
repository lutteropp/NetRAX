using Pkg
Pkg.add(PackageSpec(name="PhyloNetworks", rev="phyLiNCmanual"))

using PhyloNetworks

toy_tree_file = "/home/sarah/eclipse-workspace/NetRAX/experiments/src/toy_network.nw"

#toy_tree_file = "/home/luttersh/NetRAX/experiments/src/toy_network.nw"

gene_trees_file = ARGS[1]
start_tree_file = ARGS[2]
max_reticulations = parse(Int, ARGS[3])
n_runs = 10

# we need to do a toy inference first, as the first time we call a Julia function will be slow due to just-in-time compilation
toy_gene_trees = readMultiTopology(toy_tree_file)
toy_start_tree = readTopology(toy_tree_file)
toy_raxmlCF = readTrees2CF(toy_gene_trees, writeTab=false, writeSummary=false)
toy_net1r = snaq!(toy_start_tree, toy_raxmlCF, hmax=1, filename="snaq/toy_net1_raxml", runs=1)

# now comes the real inference run
real_gene_trees = readMultiTopology(gene_trees_file)
real_start_tree = readTopology(start_tree_file)
real_raxmlCF = readTrees2CF(real_gene_trees, writeTab=false, writeSummary=false)

real_net1r = snaq!(real_start_tree, real_raxmlCF, hmax=1, filename="snaq/net1_raxml", runs=n_runs)
real_net2r = snaq!(real_net1r, real_raxmlCF, hmax=2, filename="snaq/net2_raxml", runs=n_runs)
real_net3r = snaq!(real_net2r, real_raxmlCF, hmax=3, filename="snaq/net3_raxml", runs=n_runs)

println(real_net1r.net)
println(real_net2r.net)
println(real_net3r.net)
