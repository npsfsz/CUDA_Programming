#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
//2^29

int size = 512*1024*1024;

int* generateRandomArray(int num){
	int *result;
	time_t t;
	srand((unsigned) time(&t));
	result = (int*)malloc(sizeof(int) * num);
	for (int i = 0; i < num; i++){
		result[i] = rand() % 21 -10;
		//result[i] = 1;
	}
	return result;

}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

double CPUtime(){
       struct timeval tp;
       gettimeofday (&tp, NULL);
       return ((double)tp.tv_sec *1e6 + (double)tp.tv_usec );
}



__global__ void
reduce(int *d_iarray, int *d_oarray, int n, int blockSize){
    //__shared__ int sdata[32]; //hard coded for now hard limit 0xc000
	__device__ __shared__ int sdata[64];
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    int tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    int gridSize = blockSize*2*gridDim.x;

    //second round n = 64, blockSize = 32
    int mySum = 0;
    
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += d_iarray[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < n)
            mySum += d_iarray[i+blockSize];

        i += gridSize;
    }
    
    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();
#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) d_oarray[blockIdx.x] = mySum;

}
/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/

//copied from sample code, need modification

/*template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template<class T>
T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c = (T)0.0;

    for (int i = 1; i < size; i++)
    {
        T y = data[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}
*/
////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        //threads = 64 < 512 -> nextpow2(65/2) == 64
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
        //block4 = (64 + 127)/128
    }

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = min(maxBlocks, blocks);
    }
}


int main(){
	int iteration = 20;
	double firstR, secondR;
    printf("Starting program... \n");
    //printf("preparing array\n");
	int *h_array = generateRandomArray(size); //512M size
	//printf("array generate complete\n");
	int *d_iarray, *d_oarray;
	int blocks = 0; //the following two should be maximum
	int threads = 0;
	int maxThreads = 4096; //number of threads per block
	int maxBlocks = 1024;
	firstR = 0;
	secondR = 0;
	for (blocks = 32; blocks <= 32/*maxBlocks*/; blocks *= 2){
		for (threads = 512 ;threads <= 512/*maxThreads*/; threads *= 2){
			for (int i = 0; i < iteration; i++){
				//int bytes = sizeof(int) * size;
				//printf("%b\n", bytes);





				//d_iarray = (int*)malloc(bytes);
				//d_oarray = (int*)malloc(maxBlocks*sizeof(int));
				//alloc mem on GPU
				//int *d_array;
				//printf("copy data to GPU\n");

				cudaMalloc((void **)&d_iarray, size * sizeof(int));
				//printf("1\n");
				cudaMalloc((void **)&d_oarray, maxBlocks * sizeof(int));

				//copy data to GPU
				cudaMemcpy(d_iarray, h_array, size * sizeof(int), cudaMemcpyHostToDevice);
				//printf("copy complete\n");
				//do the work

				//getNumBlocksAndThreads(6, size, maxBlocks, maxThreads, blocks, threads);
				//define struct



				//blocks = 64;
				//threads = 256;
				dim3 grid(blocks, 1, 1);
				dim3 block(threads, 1, 1);

				//dim3 grid(64,1,1);
				//dim3 block(256,1,1);
				// when there is only one warp per block, we need to allocate two warps
				// worth of shared memory so that we don't index shared memory out of bounds
				int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

				//printf("first round of reduction\n");
				//first round of reduction
				cudaDeviceSynchronize();
				double firstStart = CPUtime();

				reduce<<< grid, block, smemSize >>>(d_iarray, d_oarray, size, threads);
				cudaDeviceSynchronize();
				double firstEnd = CPUtime();
				//printf("complete first round, took time %f\n", firstEnd-firstStart);
				firstR += firstEnd-firstStart;
				// Clear d_idata for later use as temporary buffer.
				cudaMemset(d_iarray, 0, size*sizeof(int));

				// sum partial block sums on GPU
				int s=blocks;

				//secondR = 0;
				while (s > 1)
				{
					int threads2 = 0, blocks2 = 0;
					getNumBlocksAndThreads(6, s, maxBlocks, maxThreads, blocks2, threads2);//1 block 32 threads
					cudaMemcpy(d_iarray, d_oarray, s*sizeof(int), cudaMemcpyDeviceToDevice);//prepare new input date
					//reduce<T>(s, threads, blocks, kernel, d_idata, d_odata);//reduce


					dim3 block2(threads2, 1, 1);
					dim3 grid2(blocks, 1, 1);
					//dim3 grid2(1, 1, 1);
					//dim3 block2(32, 1, 1);

					int smemSize = (threads2 <= 32) ? 2 * threads2 * sizeof(int) : threads2 * sizeof(int);
					cudaDeviceSynchronize();
					//printf("second round of reduction\n");
					double secondStart = CPUtime();
					reduce<<< grid2, block2, smemSize >>>(d_iarray, d_oarray, s, threads2);
					cudaDeviceSynchronize();
					//1 block 32 threads,
					double secondEnd = CPUtime();
					//printf("complete second round, took time %f\n", secondEnd - secondStart);
					secondR += secondEnd-secondStart;

					s = (s + (threads2*2-1)) / (threads2*2);

					/*
					if (s > 1)
					{
						// copy result from device to host
						cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost);

						for (int i=0; i < s; i++)
						{
							gpu_result += h_odata[i];
						}

						needReadBack = false;
					}
					*/
				}

				cudaFree(d_iarray);
				cudaFree(d_oarray);

			}
			// copy final sum from device to host
			double firstFinal = firstR/iteration;
			double secondFinal = secondR/iteration;
			//printf("average time, first round %f, second round %f, total %f\n", firstFinal, secondFinal, firstFinal+secondFinal);
			int gpu_result;
			cudaMemcpy(&gpu_result, d_oarray, sizeof(int), cudaMemcpyDeviceToHost);
			//printf("final GPU result is %d\n", gpu_result);
			int cpu_result = 0;
			for (int i = 0; i < size; i++){
				cpu_result += h_array[i];
			}
			//printf("final CPU result is %d\n", cpu_result);
			printf("grid %d, block %d, average time, first round %.2f, second round %.2f, total %.2f",blocks, threads, firstFinal, secondFinal, firstFinal+secondFinal);
			if (cpu_result == gpu_result){
				printf(" result matches\n");
			}else{
				printf(" result not match\n");
			}
			cudaFree(d_iarray);
			cudaFree(d_oarray);
			cudaDeviceSynchronize();
		}
	}

	return 0;
}


