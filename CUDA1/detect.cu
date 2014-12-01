#include "detect.h"

int detectBestGPU(){
     cudaDeviceProp prop;
     //cudaError_t ret;
     int nb_devices=1;
     int best=0;
     int max_count=0;
     cudaGetDeviceCount((&nb_devices));
     for (int i =0; i<nb_devices;i++){
	cudaGetDeviceProperties ( (&prop), i );
	if (prop.multiProcessorCount>max_count){
		best=i;
		max_count=prop.multiProcessorCount;
        }
     }
     return best;
}

cudaError_t setBestGPU()
{
    int device = detectBestGPU();
    cudaDeviceProp prop;
    cudaGetDeviceProperties ( (&prop), device );
    cudaError_t ret = cudaSetDevice ( device );
    printf("%s has been choosed\n",prop.name); 
    return ret;
}



