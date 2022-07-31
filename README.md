[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/lutteropp/NetRAX) 

# NetRAX
Phylogenetic Network Inference without ILS

1. **Install the dependencies.** On Ubuntu (and other Debian-based systems), you can simply run:
```
sudo apt-get install flex bison libgmp3-dev cmake doxygen libmpfrc++-dev libopenmpi-dev
```

2. **Build instructions**
(Tested on Ubuntu with GCC)
```
git clone --recurse-submodules https://github.com/lutteropp/NetRAX.git
cd NetRAX
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON ..
make
```

3. **Usage Examples**

For a detailed list of NetRAX commands, run:
```
./netrax --help
```

Run a NetRAX network inference, starting from a single start network (or tree), using LhModel.AVERAGE:
```
mpiexec ./netrax --msa example_msa.fasta --model example_partitions.txt --average_displayed_tree_variant --start_network my_start_network.nw --output my_inferred_network.txt --seed 42
```

Judge a NetRAX inference result using LhModel.BEST, computing normalized topological network distances and BIC:
```
mpiexec ./netrax --msa example_msa.fasta --model example_partitions.txt --best_displayed_tree_variant --start_network my_inferred_network.nw --judge my_true_network.nw --judge_only
```

4. **Convenience Python Wrappers**

For the following two Python wrappers, ensure that the absolute paths to NetRAX and RAxML-NG are set correctly in the Python source files.

Generate a set of start trees, using RAxML-NG:
```
python3 build_start_trees.py --name example_start_trees --msa_path example_msa.fasta --partitions_path example_partitions.txt --num_parsimony_trees 10 --num_random_trees 10 --seed 42
```

Run a NetRAX network inference using the more user-friendly Python wrapper, starting from a set of user-specified start networks:
```
python3 netrax.py --name example --msa_path example_msa.fasta --partitions_path example_partitions.txt --likelihood_type average --start_networks my_start_networks.txt --seed 42
```

Can't open your network inferred by NetRAX with the Dendroscope tool? Worry not! This is because Dendroscope does not work with support values or reticulation probablities being present in the Extended NEWICK file. You can use this script to get rid of these:
```
python3 netrax_output_to_dendroscope.py my_netrax_network.txt > network_for_dendroscope.txt
```

<!---
3. **Documentation**
To build the documentation, run ```doxygen Doxyfile```, and then open the generated docs/html/index.html in your browser.

4. **Testing**
To run the tests, run ```./test/bin/netrax_test``` from the root NetRAX project folder.
-->
