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

    GpuTimer timer;
    timer.Start();

    synthesis<<<1, num_of_iterations>>>();

    timer.Stop();
    printf("*** Time Elapsed: %f ms ***\n", timer.Elapsed());

    // TODO - Free memory

    return 0;
}
