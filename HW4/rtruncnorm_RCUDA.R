library(RCUDA)
library(xtable)

cuGetContext(TRUE)

m = loadModule("rtruncnorm.ptx")

k = m$rtruncnorm_kernel

tk = 1:8
RunTime=matrix(0,8,10)
Result=matrix(0,8,2)
for (i in 1:length(tk)){
  N = as.integer(10^tk[i])
  vals = rep(0,N)
  mu = rep(2, N)
  sigma = rep(1, N)
  lower = rep(-Inf, N)
  high = rep(-10, N)
  mu_length = N
  sigma_length = N
  low_length = N
  high_length = N
  rng_seed_a = 1234L
  rng_seed_b = 1423L
  rng_seed_c = 1842L
  maxtries = 2000L
  
#Dim @ k=8
  threads_per_block_8 <- 512L
  block_dims_8 <- c(as.integer(threads_per_block_8), 1L, 1L)
  grid_d1_8 <- as.integer(floor(sqrt(N/threads_per_block_8)))
  grid_d2_8 <- as.integer(ceiling(N/(grid_d1_8*threads_per_block_8)))
  grid_dims_8 <- c(grid_d1_8, grid_d2_8, 1L)
  "compute_grid" <- function(N,sqrt_threads_per_block=16L,grid_nd=1)
  {
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
  #When k=1,...,7,use the compute_grid function,when k=8, take another dimension.
  if (i==8){
  grid_dims=grid_dims_8
  block_dims=block_dims_8
  }else{
  grid_dims = grid$grid_dims
  block_dims = grid$block_dims
  }
  print(grid_dims)

  print(block_dims)
  
  nthreads <- prod(grid_dims)*prod(block_dims)
  cat("Total number of threads = ",nthreads,"\n \n")
  if (nthreads < N){
    stop("Grid is not large enough...!")
  }

    copy_to_time <- system.time({
      vals_dev = copyToDevice(vals)
      mu_dev = copyToDevice(mu)
      sigma_dev = copyToDevice(sigma)
      hi_dev = copyToDevice(hi)
      lo_dev = copyToDevice(lo)
    })
   RunTime[i,1:3]=as.vector(copy_to_time[1:3])
    print(copy_to_time)
    cat("done. Copying to device...\n \n")
    
    kernel_time <- system.time({
      .cuda(k, vals_dev, N, mu_dev, sigma_dev, lo_dev, hi_dev, mu_len, sigma_len, 
            lo_len, hi_len, rng_seed_a, rng_seed_b, rng_seed_c, maxtries, 
            gridDim = grid_dims, blockDim = block_dims)
    })
   RunTime[i,4:6]=as.vector(kernel_time[1:3])
    print(kernel_time)
    
    copy_from_time <- system.time({
      vals = copyFromDevice(obj=vals_dev,nels=vals_dev@nels,type="float")
    })
   RunTime[i,7:9]=as.vector(copy_from_time[1:3])
    print(copy_from_time)

  RunTime[i,10] = sum(Time[i,3],RunTime[i,6],RunTime[i,9])
  Result.observed = Results(vals)
  
  print(summary.observed)

   mean_theo = round(mu[1] -
      dnorm((hi[1]-mu[1]))/(pnorm((hi[1]-mu[1])/sigma[1])), 5)

  
  mean_obs = round(Results.observed["Mean"], 6)
  sd_obs = round(sd(vals),6)
  Results[i,] = c(mean_obs,sd_obs)
  label_theo= paste0("Theoretical Mean = ", mean_theo)
  label_obs = paste0("Observed Mean = ", mean_obs)
  
  png("density.png")
    plot(density(vals), main="Density of Upper Truncation, TN(2,1;(-Inft,-10))", lwd=1.5)
    abline(v=mean_theo, col="red", lwd=2)
    abline(v=mean_obs, lty="dashed", col="green",lwd=2)
    legend("topleft", inset=.05, legend=c(label_theo, label_obs), lty=c("solid", "dashed"), lwd=c(2,2), col=c("red","green"))
  dev.off()  
}
xtable(RunTime,digits=4)
xtable(Results,digits=4)


rm(list=ls())

q("no")