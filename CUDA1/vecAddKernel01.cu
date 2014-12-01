

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
	int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = blockStartIndex + (threadIdx.x * N);
    int threadEndIndex   = threadStartIndex + N;
    int i;
    int add;
    printf("hello\n");
    for( i=threadStartIndex; i<threadEndIndex; i++ ){
    	add = ((i - threadStartIndex)*blockDim.x) + threadStartIndex;
		C[add] = A[add] + B[add];
    }
}
