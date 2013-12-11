#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

extern "C" 
{

	__device__ float rand_expon(float a, curandState *state)
	{
		return -log(curand_uniform(state))/a;  // x is now random expon by inverse CDF
	} // END rand_expo


	__device__ float psi_calc(float mu_minus, float alpha, float z)
	{
		float psi;
        	// Compute Psi 
                if(mu_minus < alpha){
                	psi = expf( -1/2*pow(alpha-z,2));
                }
                else {
                        psi = expf(  1/2*( pow(mu_minus-alpha,2) - pow(alpha-z,2) ) );
                }
		return psi;
	}

	__global__ void rtruncnorm_kernel(float *vals, int n, 
        	float *mu, float *sigma, 
                float *lo, float *hi,
                int mu_len, int sigma_len,
                int lo_len, int hi_len,
		int rng_seed_a, int rng_seed_b, int rng_seed_c,
                int maxtries)
	{
    		int accepted = 0;
    		int numtries = 0;
		float x;
    		float u;
    		float alpha;
		float psi;
    		float z;
    		float a;
    		float mu_minus;
    		int left_trunc = 0;

	    	// Figure out which thread and block you are in and map these to a single index, "idx"
    		// Usual block/thread indexing...
    		int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    		int blocksize = blockDim.x * blockDim.y * blockDim.z;
    		int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    		int idx = myblock * blocksize + subthread;
    
    		// Check: if index idx < n generate a sample, else in unneeded thread 
    		if(idx<n){

    	    		// Setup the RNG:
	    		curandState rng;
	    		curand_init(rng_seed_a + idx*rng_seed_b, rng_seed_c, 0, &rng);

	   	    	// Sample the truncated normal
	    		// i.e. pick off mu and sigma corresponding to idx and generate a random sample, x
	    		// if that random sample, x, is in the truncation region, update the return value to x, i.e. vals[idx]=x
	    		// if x is not in the trunc region, try again until you get a sample in the trunc region or if more than maxtries,
	    		// move on to Robert's approx method
	    		while(accepted == 0 && numtries < maxtries){
				numtries++;  // Increment numtries
				x = mu[idx] + sigma[idx]*curand_normal(&rng);
				if(x >= lo[idx] && x <= hi[idx]){
					accepted = 1;
					vals[idx] = x;
				}
	    		} 

	    		// Robert's approx method
	    		// We don't want to write both trunc algos for left and right tail truncations, just use 
			// right tail trancation.  If we want to sample from Y~N(mu, sigma, -Inf, b), we transform 
			// first X~N(mu, sigma, -b+2*mu, Inf), use only right truncation, sample from the right 
			// tail to get a X, then transform back Y=2*mu-X to get left truncation sample if needed in Robert.  
		    	if(lo[idx] < mu[idx]) {			// then left truncation
				left_trunc = 1;
				a = -1*hi[idx] + 2*mu[idx];		// flip up to right tail  
	    		}
	    		else {
				a = lo[idx];				// right truncation from a=lo[idx] to infinity
	    		}	
		    	mu_minus = (a-mu[idx])/sigma[idx];

            		// need to find mu_minus but that depends on if lower trunc or upper trunc
	            	alpha = (mu_minus + sqrtf(pow(mu_minus,2) + 4))/2;
			numtries = 1;	//  If couldn't get sample naively, reset and try Robert
	    		while(accepted == 0 && numtries < maxtries){
				
				numtries++;  // Increment numtries

				// Need random expon for Robert no curand_expon function so do inverse CDF
				// F(x) = 1-exp(-alpha*x) --> F^1(x) = -log(U)/alpha where U~Unif[0,1]
				// u = curand_uniform(&rng);
				// x = -1 * log(u)/alpha;  // x is now random expon by inverse CDF 
				z = mu_minus + rand_expon(alpha, &rng);

				// Compute Psi = probability of acceptance
				psi = psi_calc(mu_minus, alpha, z);

				// Check if Random Unif[0,1] < Psi, if so accept, else reject and try again
				u = curand_uniform(&rng);
				if (u < psi){
					accepted = 1;	// we now have our vals[idx]
					if (left_trunc == 1){  // since originally left trunc, and flip back to left tail and final transform
						vals[idx] = mu[idx] - sigma[idx]*z;
					}
					else {   // right truncation originally so we're done after final transform
						vals[idx] = mu[idx] + sigma[idx]*z;
					}
				}
	    		}
            		if(accepted == 0){	// Just in case both naive and Roberts fail
	            		vals[idx] = -999;
            		}

    		} // END if (idx<n)
    		return;
	} // END rtruncnorm_kernel
} // END extern "C"
