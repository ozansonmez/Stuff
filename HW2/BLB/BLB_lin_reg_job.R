##Bag of Little BootStraps

s = 5           # s = number of samples (of size b)
r = 50          # r = number of bootstrap replicates per subset

###############################===Setup for running on Gauss===########################

args <- commandArgs(TRUE)

cat("Command-line arguments:\n")
print(args)

####
# sim_start ==> Lowest possible dataset number
###

###################
sim_start <- 1000
###################

if (length(args)==0){
  sim_num <- sim_start + 1
  set.seed(121231)
} else {
  # SLURM can use either 0- or 1-indexing...
  # Lets use 1-indexing here...
  sim_num <- sim_start + as.numeric(args[1])
  
  
  
  
  # Get the job number from the argument 1:(s*r) (i.e. 1:5*50 = 1:250)
  job = as.numeric(args[1])
  
  ##Need to get the s and r index
  # Get the s_index by using mod s.  
  # Also, if job mod s == 0, then it is subsample dataset s (i.e. 5)
  s_index = job %% s
  if (s_index == 0){
    s_index = s
  }
  
  # Get r_index 
  # Also, if job mod r == 0, then it is bootstrap sample r (i.e. 50) within subsample dataset s
  r_index = ceiling(job / s)
  if (r_index == 0){
    r_index = r
  }
  
  # The seed must be a function of the s_index to ensure that the subsample is the same
  # for same values of s (Thanks MBissel for your help with this)
  sim_seed <- (762*(s_index) + 121231)
  set.seed(sim_seed)
}


cat(paste("\nAnalyzing dataset number ",sim_num,"...\n\n",sep=""))





####################===== BLB Algorithm =====#######################


library(BH)
library(bigmemory)
library(bigmemory.sri)
library(biganalytics)



#Set path names

datapath <- "/home/pdbaines/data/"
outpath <- "output/"
rootfilename <- "blb_lin_reg_data"
datafile = paste0(datapath, rootfilename, ".desc")


#get the data as a big matrix (1,000,000x1,001) where the last column
#is theresponse variable. Number of rows is the size ofthe data n, and the 
#number of covariates is the # of columns minus 1 since the last col is y.

#set the parameters for BLB
data<-attach.big.matrix(datafile)
n=dim(data)[1]
gamma=0.7
b=ceiling(n^(gamma))
d=dim(data)[2]-1


#We will subsample b samples from n original distinct data points. To do so,we
#sample b elements from 1,..,n to create index set and we will use the index set 
#to determine what to sample from original data matrix.
Index.sample = sample(1:n, size=b, replace=FALSE)
sample.X = data[Index.sample,1:d]
sample.Y = data[Index.sample,d+1]

# Reset the simulation seed, set the seed for the r bootstrap samples
#We run r Bootstrap within s subsample and the seed for bootstrap sample must be 
#different for each bootsrtap sample r. Simulation seed is a function of s_index and r_index
sim_seed <- (762*(s_index) + 121231 + r_index)
set.seed(sim_seed)


# Resample n data points from subsample of size b (b<n) with replacement.
# To sample n points form b we use rmultinom 
weights<-rmultinom(1, size = n, prob=rep(1/b, b))

# Fit the linear regression and get the coefficients. Apply the weights.
# Make sure to account for no intercept term.
model = lm(sample.Y ~ 0 + sample.X, weights=weights)
beta.hat = model$coefficients

# Output file:
outfile = paste0("output/","coef_",sprintf("%02d",s_index),"_",sprintf("%02d",r_index),".txt")

# Save estimates to file:
write.table(x=beta.hat,file=outfile, sep=",", col.names=TRUE, row.names=FALSE)

q("no")
