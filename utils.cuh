#ifndef UTILS_H
#define UTILS_H

#define CUDA_CALL(x)                                        \
    do                                                      \
    {                                                       \
        if ((x) != cudaSuccess)                             \
        {                                                   \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            return EXIT_FAILURE;                            \
        }                                                   \
    } while (0)

void startTiming(cudaEvent_t *start, cudaEvent_t *stop);
float stopTiming(cudaEvent_t *start, cudaEvent_t *stop);

#endif // UTILS_H
