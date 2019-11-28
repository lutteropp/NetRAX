#Time is measured by expected substitutions per site throughout the network simulations 
#so that \theta=  N \mu is used for all population sizes and \tau_i=t_i \mu til for the
#time of node i. 
#The substitution rate \mu is fixed to 1.0 across all
#gene lineages (strict molecular clock) and all loci (no rate variation).
#We could use partitions with different rates in seqgen to have rate variation.
#Do we want to loose the strict molecular clock thing?

# arg 1 is the basename for the names for the trees (${output}_trees), network (${output}_network) and alignement.dat
# arg 2 is the number of trees to be extracted

output=$1
nb_sites=$2
	
seqGenPATH="/Users/celinescornavacca/Documents/devel/develFromOthers/Seq-Gen-1.3.4/source/seq-gen";

#egenerate the network and the trees
python3 RandomNetworkandTrees.py -nb_tree ${nb_sites} -no_ILS

#generate one character per tree via seqGen, please change the model if needed 
#since seqGen is used, no indel are simulated
${seqGenPATH} -mHKY -t3.0 -f0.3,0.2,0.2,0.3 -l${nb_sites} -p${nb_sites}< ${output}_trees > ${output}.dat