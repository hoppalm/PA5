#ifndef _DETECT_H
#define _DETECT_H

#include <stdio.h>

int detectBestGPU(); // find the GPU with the highest processor count.
cudaError_t setBestGPU(); // use the previous function to choose and set the best GPU.

#endif
