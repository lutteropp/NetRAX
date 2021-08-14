#Load internal functions
source('./Functions.R')

start_time <- Sys.time()

seqs = read.nexus.data('10t_1r/0_0.nex')

res = SSEM(seqs)
print(res$logLik) #print out the Maximum log-likelihood

#Plot the infer structure
plot(res[[1]], attrs=list(node=list(shape='plaintext', fixedsize=F)))

end_time <- Sys.time()
elapsed_time <- end_time - start_time
print(elapsed_time)