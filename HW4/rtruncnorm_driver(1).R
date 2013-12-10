library(RCUDA)
library(xtable)
cat("\nSetting cuGetContext(TRUE)...\n ")
cuGetContext(TRUE)
cat("done. Profiling CUDA code.\n \n")

cat("Loading module...\n")
m = loadModule("rtruncnorm.ptx")
cat("done. Loading module.\n \n")

cat("Extracting kernelkernel...\n")
k = m$rtruncnorm_kernel
cat("done. Extracting kernelkernel.\n \n")

cat("Setting up input params...\n")


t_k = 1:8
TimeMatrix=matrix(0,8,10)
SummaryMatrix=matrix(0,8,2)
for (i in 1:length(t_k)){
  N = as.integer(10^t_k[i])
  
  vals = rep(0,N)
  mu = rep(2, N)
  sigma = rep(1, N)
  lo = rep(-Inf, N)
  hi = rep(-10, N)
  mu_len = N
  sigma_len = N
  lo_len = N
  hi_len = N
  rng_seed_a = 1234L
  rng_seed_b = 1423L
  rng_seed_c = 1842L
  
  maxtries = 2000L
  cat("done. Setting input params.\n \n")
  
  # Fix block dims:
  threads_per_block8 <- 512L
  block_dims8 <- c(as.integer(threads_per_block8), 1L, 1L)
  grid_d1_8 <- as.integer(floor(sqrt(N/threads_per_block8)))
  grid_d2_8 <- as.integer(ceiling(N/(grid_d1_8*threads_per_block8)))
  grid_dims8 <- c(grid_d1_8, grid_d2_8, 1L)
  
  "compute_grid" <- function(N,sqrt_threads_per_block=16L,grid_nd=1)
  {
    # if...
    # N = 1,000,000
    # => 1954 blocks of 512 threads will suffice
    # => (62 x 32) grid, (512 x 1 x 1) blocks
    # Fix block dims:
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
  if (i==8){
  grid_dims=grid_dims8
  block_dims=block_dims8
  }else{
  grid_dims = grid$grid_dims
  block_dims = grid$block_dims
  }

  cat("Grid size:\n")
  print(grid_dims)
  cat("Block size:\n")
  print(block_dims)
  
  nthreads <- prod(grid_dims)*prod(block_dims)
  cat("Total number of threads to launch = ",nthreads,"\n \n")
  if (nthreads < N){
    stop("Grid is not large enough...!")
  }
  
  cat("*********************** \n")
  cat(paste(" SYSTEM TIMES for N = 10^", t_k[i], sep=""), "\n")
  cat("*********************** \n")
  
  cat("Running CUDA kernel...\n \n")

    copy_to_time <- system.time({
      cat("Copying to device...\n")
      vals_dev = copyToDevice(vals)
      mu_dev = copyToDevice(mu)
      sigma_dev = copyToDevice(sigma)
      hi_dev = copyToDevice(hi)
      lo_dev = copyToDevice(lo)
    })
   TimeMatrix[i,1:3]=as.vector(copy_to_time[1:3])
    print(copy_to_time)
    cat("done. Copying to device...\n \n")
    
    kernel_time <- system.time({
      cat("Call the kernel...\n")
      .cuda(k, vals_dev, N, mu_dev, sigma_dev, lo_dev, hi_dev, mu_len, sigma_len, 
            lo_len, hi_len, rng_seed_a, rng_seed_b, rng_seed_c, maxtries, 
            gridDim = grid_dims, blockDim = block_dims)
    })
   TimeMatrix[i,4:6]=as.vector(kernel_time[1:3])
    print(kernel_time)
    cat("done. Calling the kernel...\n \n")
    
    copy_from_time <- system.time({
      cat("Copying result back from device...\n")
      vals = copyFromDevice(obj=vals_dev,nels=vals_dev@nels,type="float")
    })
   TimeMatrix[i,7:9]=as.vector(copy_from_time[1:3])
    print(copy_from_time)
    cat("done. Copying result back from device...\n \n")

  cat("done. Running CUDA kernel.\n \n")
  TimeMatrix[i,10] = sum(TimeMatrix[i,3],TimeMatrix[i,6],TimeMatrix[i,9])
  summary.obs = summary(vals)
  
  print(summary.obs)
  
#   # Two sided truncation
#   mean_theoretical = round(mu[1] + 
#     sigma[1]*(dnorm((lo[1]-mu[1])/sigma[1])-dnorm((hi[1]-mu[1])/sigma[1]))/
#     (pnorm((hi[1]-mu[1])/sigma[1])-pnorm((lo[1]-mu[1])/sigma[1])), 5)
  
#  # Upper truncation
  mean_theoretical = round(mu[1] -
      dnorm((hi[1]-mu[1]))/(pnorm((hi[1]-mu[1])/sigma[1])), 5)
#   
# #   # Lower truncation
#   mean_theoretical = round(mu[1] + dnorm((lo[1]-mu[1]))/(1-pnorm(lo[1]-mu[1])),5)
  
  mean_obs = round(summary.obs["Mean"], 6)
  sd_obs = round(sd(vals),6)
  SummaryMatrix[i,] = c(mean_obs,sd_obs)
  label_theoretical= paste0("Theoretical Mean = ", mean_theoretical)
  label_observed = paste0("Observed Mean = ", mean_obs)
  
  pdf("density.pdf")
    plot(density(vals), main="Density of Upper Truncated, Lower Tail Normals on GPU, TN(2,1;(-Infinity,-10))", lwd=1.5)
    abline(v=mean_theoretical, col="blue", lwd=2)
    abline(v=mean_obs, lty="dashed", col="red",lwd=2)
    abline(h=0)
    legend("topright", inset=.05, legend=c(label_theoretical, label_observed), lty=c("solid", "dashed"), lwd=c(2,2), col=c("blue","red"))
  dev.off()  
}
xtable(TimeMatrix,digits=5)
xtable(SummaryMatrix,digits=5)


rm(list=ls())

q("no")