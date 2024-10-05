#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include "utils.cuh"

#define THREAD_BLOCK_SIZE 128
#define MAX_ROLLS 1000000000

__device__ unsigned int won = 0;
__device__ volatile uint8_t volatile_won = 0;
__device__ unsigned int max_ones = 0;
__device__ unsigned long long int rolls = 0;

// Sets up a random state for each thread that will attempt rolls
__global__ void setupKernel(curandState *state, uint32_t time_seed)
{
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(time_seed, threadIdx.x, 0, &state[threadIdx.x]);
}

// Attempts a roll
__global__ void runRoll(curandState *state, unsigned long long int max_rolls)
{
    /* Skip running if a winning roll was encountered */
    if (volatile_won)
        return;

    unsigned long long int roll_number = blockDim.x * blockIdx.x + threadIdx.x;
    
    /* Avoid running outside the interval [0, max_rolls) */
    if (roll_number >= max_rolls)
        return;

    int count = 0;
    unsigned int x;

    /* Copy state to local memory for efficiency */
    curandState local_state = state[threadIdx.x];

    /* Generate pseudo-random unsigned ints */
    for (int i = 0; i < 231; ++i) {
        x = curand(&local_state) & 0b11; // modulo 4 with bits (rand value from 0 to 3)
        /* Check if obtained value is zero (a paralysis proc) */
        if (!x)
            ++count;
    }

    /* Copy state back to global memory */
    state[threadIdx.x] = local_state;

    /* Update the global max_ones and rolls counts */
    atomicMax(&max_ones, count);
    atomicAdd(&rolls, 1);

    /* Check if a winning roll took place */
    if (count >= 177) {
        /* Mark that a thread won */
        unsigned int prev_value = atomicCAS(&won, 0U, 1U);

        /* Check if no other thread found a winning roll before the current one */
        if (prev_value == 0) {
            /* Signal other threads to stop attempting rolls */
            volatile_won = 1;
        }
    }
}

int main(int argc, char *argv[])
{
    curandState *states;
    unsigned int result_max_ones;
    unsigned long long int total_rolls, max_rolls = MAX_ROLLS;

    cudaEvent_t start, stop;
    startTiming(&start, &stop);

    /* Check if alternative rolls number provided as argument */
    if (argc > 1) {
        sscanf(argv[1], "%llu", &max_rolls);
    }

    /* Allocate space for pseudo-random generator states on the GPU */
    CUDA_CALL(cudaMalloc((void **)&states, THREAD_BLOCK_SIZE * sizeof(curandState)));

    /* Setup pseudo-random generator states for every thread */
    setupKernel<<<1, THREAD_BLOCK_SIZE>>>(states, time(NULL));

    /* Simulate all rolls */
    unsigned long long int blocks_no = (max_rolls + THREAD_BLOCK_SIZE) / THREAD_BLOCK_SIZE;
    runRoll<<<blocks_no, THREAD_BLOCK_SIZE>>>(states, max_rolls);

    /* Get the max number of procs rolled in an attempt from GPU */
    CUDA_CALL(cudaMemcpyFromSymbol(&result_max_ones, max_ones, sizeof(max_ones)));

    /* Get the number of rolls attempted from the GPU */
    CUDA_CALL(cudaMemcpyFromSymbol(&total_rolls, rolls, sizeof(rolls)));

    /* Cleanup */
    CUDA_CALL(cudaFree(states));

    /* Get elapsed time */
    float seconds = stopTiming(&start, &stop);

    /* Print results */
    printf("Highest Ones Roll: %u\n", result_max_ones);
    printf("Number of Roll Sessions: %llu\n", total_rolls);
    printf("Time Taken: %f seconds\n", seconds);
    printf("\n");

    return EXIT_SUCCESS;
}