#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "utils.cuh"

void startTiming(cudaEvent_t *start, cudaEvent_t *stop) {
    cudaEventCreate(start);
    cudaEventCreate(stop);
    cudaEventRecord(*start);
}

float stopTiming(cudaEvent_t *start, cudaEvent_t *stop) {
    cudaEventRecord(*stop);
    cudaEventSynchronize(*stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, *start, *stop);
    cudaEventDestroy(*start);
    cudaEventDestroy(*stop);
    return milliseconds / 1000.0f;
}
