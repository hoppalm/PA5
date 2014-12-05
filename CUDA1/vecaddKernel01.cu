///
/// vecAddKernel01.cu
///
/// This Kernel adds two Vectors A and B in C on GPU
/// with using coalesced memory access.
/// 

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
	int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = blockStartIndex + threadIdx.x;
    int threadEndIndex   = threadStartIndex + N;
    int i, add;

    //loop n times coalescing through vectors
    for( i=threadStartIndex; i<threadEndIndex; ++i){
        //get the index to add from the two vectors, jumps across vector each iteration
        //not one index a time though
    	add = ((i - threadStartIndex)*blockDim.x) + threadStartIndex;
		C[add] = A[add] + B[add];
    }
}