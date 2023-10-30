#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "gputimer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void synthesis() {
    // TODO - implement synthesis
}

int main(int argc, char** argv) {
    int num_of_iterations = atoi(argv[1]);

    // TODO - Define the number of blocks and threads per block
    // Allocate memory
    float *u = (float*) malloc(sizeof(float) * SIZE * SIZE);
    float *u1 = (float*) malloc(sizeof(float) * SIZE * SIZE);
    float *u2 = (float*) malloc(sizeof(float) * SIZE * SIZE);

    float *result = (float*) malloc(sizeof(float) * num_of_iterations);

    memcpy(u1, u, sizeof(float) * SIZE * SIZE);
    memcpy(u2, u, sizeof(float) * SIZE * SIZE);

    GpuTimer timer;
    timer.Start();

    synthesis<<<1, num_of_iterations>>>();

    timer.Stop();
    printf("*** Time Elapsed: %f ms ***\n", timer.Elapsed());

    // TODO - Free memory

    return 0;
}
