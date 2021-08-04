using Pkg
Pkg.add(PackageSpec(name="PhyloNetworks", rev="phyLiNCmanual"))

using PhyloNetworks

toy_network_file = "/home/sarah/code-workspace/NetRAX/experiments/src/toy_network.nw"
toy_msa_file = "/home/sarah/code-workspace/NetRAX/experiments/src/toy_msa.fasta"

#toy_network_file = "/home/luttersh/NetRAX/experiments/src/toy_network.nw"
#toy_msa_file = "/home/luttersh/NetRAX/experiments/src/toy_msa.fasta"

start_network_file = ARGS[1]
msa_file = ARGS[2]
max_reticulations = parse(Int, ARGS[3])
n_runs = 10

# we need to do a toy inference first, as the first time we call a Julia function will be slow due to just-in-time compilation
toy_start_network = readTopology(toy_network_file)
toy_result = phyLiNC(toy_start_network, toy_msa_file, :HKY85, maxhybrid=max_reticulations, seed=42, nruns=1)

# now comes the real inference run
real_start_network = readTopology(start_network_file)
real_result = phyLiNC(real_start_network, msa_file, :HKY85, maxhybrid=max_reticulations, seed=42, nruns=n_runs)
println(real_result.net)