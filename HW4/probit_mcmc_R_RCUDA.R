#Hw 4 Question 2
#setwd("~/GitHub/Stuff/HW4")
library(RCUDA)
library(mvtnorm)
library(truncnorm)

#CPU Code
#Load data
data1 = read.table("data_01.txt", header=T, quote="\"")
data2 = read.table("data_02.txt", header=T, quote="\"")
data3 = read.table("data_03.txt", header=T, quote="\"")
data4 = read.table("data_04.txt", header=T, quote="\"")
data5 = read.table("data_05.txt", header=T, quote="\"")
datamini = read.table("mini_data.txt", header=T, quote="\"")

pars1 = read.table("pars_01.txt", header=T, quote="\"")
pars2 = read.table("pars_02.txt", header=T, quote="\"")
pars3 = read.table("pars_03.txt", header=T, quote="\"")
pars4 = read.table("pars_04.txt", header=T, quote="\"")
pars5 = read.table("pars_05.txt", header=T, quote="\"")

#Adjust them based on which data set is being applied
data = data3 
pars = pars3

y = as.matrix(data[,1])
x = as.matrix(data[,-1])
p = ncol(x)
beta_0 = matrix(0,p,1)
Sigma_0_inv = matrix(1,p,p)

probit_mcmc_cpu = function(y,x,beta_0,Sigma_0_inv,niter,burnin){
  n = nrow(x)
  beta.matrix = matrix(0,burnin+niter,p)
  beta.t = beta_0 
  for (i in 1:(burnin+niter)){
  #Generate z
  z = ifelse(y>0,rtruncnorm(1,0,Inf,x%*%beta.t,1),rtruncnorm(1,-Inf,0,x%*%beta.t,1))
  # Adjust the posterior parameters
  Sigma.t = solve(Sigma_0_inv + t(x)%*%x) 
  mu.t = Sigma.t%*%(t(x)%*%as.matrix(z)+Sigma_0_inv%*%beta_0) 
  # Sample from the posterior
  t(rmvnorm(1,mu.t,Sigma.t)) 
  # Save betas
  beta.matrix[i,] = beta.t
  } 
  return(colMeans(beta.matrix[(burnin+1):(burnin+niter),]))
} 

### GPU #############
cuGetContext(TRUE)

m = loadModule("rtruncnorm.ptx")

k = m$rtruncnorm_kernel

######################

probit_mcmc_gpu = function(y,x,beta_0,Sigma_0_inv,niter,burnin){
  
  p = ncol(x)
  N = as.integer(nrow(x))
  
  vals = rep(0,N)
  beta.matrix = matrix(0,nrow=(burnin+niter), ncol = p)
  beta.t = beta_0 
  lo = ifelse(y>0,0,-Inf)
  hi = ifelse(y>0,Inf,0)
  sigma = matrix(1,nrow=N, ncol=1)  
  mu_len = N
  sigma_len = N
  lo_len = N
  hi_len = N
  rng_seed_a = 1234L
  rng_seed_b = 1423L
  rng_seed_c = 1842L
  maxtries = 2000L
  
  "compute_grid" <- function(N,sqrt_threads_per_block=16L,grid_nd=1){
    
    block_dims <- c(as.integer(sqrt_threads_per_block), as.integer(sqrt_threads_per_block), 1L)
    threads_per_block <- prod(block_dims)
    if (grid_nd==1){
      grid_d1 <- as.integer(max(1L,ceiling(N/threads_per_block)))
      grid_d2 <- 1L
    } else {
      grid_d1 <- as.integer(max(1L, floor(sqrt(N/threads_per_block))))
      grid_d2 <- as.integer(ceiling(N/(grid_d1*threads_per_block)))
    }
    grid_dims <- c(grid_d1, grid_d2, 1L)
    return(list("grid_dims"=grid_dims,"block_dims"=block_dims))
  }
  grid = compute_grid(N)
  grid_dims = grid$grid_dims
  block_dims = grid$block_dims
  
  print(grid_dims)
  print(block_dims)
  
  nthreads <- prod(grid_dims)*prod(block_dims)
  cat("Total number of threads = ",nthreads,"\n \n")
  if (nthreads < N){
    stop("Grid is not large enough!")
  }
  cat("Copying to device...\n")
  vals_dev = copyToDevice(vals)
  sigma_dev = copyToDevice(sigma)
  hi_dev = copyToDevice(hi)
  lo_dev = copyToDevice(lo)
  
  for (i in 1:(burnin+niter)){
    
    mu.t = x%*%beta.t  
    mu_dev = copyToDevice(mu.t)  
    
        .cuda(k, vals_dev, N, mu_dev, sigma_dev, lo_dev, hi_dev, mu_len, sigma_len, 
          lo_len, hi_len, rng_seed_a, rng_seed_b, rng_seed_c, maxtries, 
          gridDim = grid_dims, blockDim = block_dims)
    
    vals = copyFromDevice(obj=vals_dev,nels=vals_dev@nels,type="float")      

   
    sigma.post= solve(Sigma_0_inv + t(x)%*%x) 
    mu.post = sigma.post%*%(t(x)%*%as.matrix(vals)+Sigma_0_inv%*%beta_0)
    
    beta.t = t(rmvnorm(1,mu.post,sigma.post))
    beta.matrix[i,] = beta.t
  } 
  
  return(colMeans(beta.matrix[(burnin+1):(burnin+niter),]))
} 



#Run CPU and GPU in AWS to get the runtimes and means of betas
time.cpu<-system.time({betaMeans_CPU = probit_mcmc_cpu(y = y, x = x, beta_0 = beta_0, Sigma_0_inv = Sigma_0_inv, niter = 2000, burnin = 500)})
write.table(rbind(round(t(betaMeans_CPU),6),round(t(pars),6),round(t(betaMeans_CPU - pars),6)),"Beta Means CPU",sep=" ",col.names=FALSE, row.names=FALSE)
write.table(round(t(as.vector(time.cpu[1:3])),4),"time_CPU",sep=" ",col.names=FALSE,row.names=FALSE)           

time.gpu<-system.time({betaMeans_GPU = probit_mcmc_gpu(y = y, x = x, beta_0 = beta_0, Sigma_0_inv = Sigma_0_inv, niter = 2000, burnin = 500)})
write.table(rbind(round(t(betaMeans_GPU),6),round(t(pars),6),round(t(betaMeans_GPU - pars),6)),"Beta Means GPU",sep=" ",col.names=FALSE, row.names=FALSE)
write.table(round(t(as.vector(time.gpu[1:3])),4),"time_GPU",sep=" ",col.names=FALSE,row.names=FALSE) 