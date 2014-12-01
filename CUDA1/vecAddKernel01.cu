

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = blockStartIndex + (threadIdx.x * N);
    int threadEndIndex   = threadStartIndex + N;
    int i;

    for( i=threadStartIndex; i<threadEndIndex; ++i ){
		C[threadStartIndex + (blockDim.x * i)] = A[threadStartIndex + (blockDim.x * i)] + B[threadStartIndex + (blockDim.x * i)];
    }
}
