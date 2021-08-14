#Load internal functions
source('./Functions.R')

start_time <- Sys.time()

#Please replace with your nexus sequence file
seqs = read.nexus.data('20t_2r/20t_2r.nex')

res = SSEM(seqs)
print(res$logLik) #print out the Maximum log-likelihood

#Plot the infer structure
plot(res[[1]], attrs=list(node=list(shape='plaintext', fixedsize=F)))

end_time <- Sys.time()
elapsed_time <- end_time - start_time
print(elapsed_time)