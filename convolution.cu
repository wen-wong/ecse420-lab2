#include <stdio.h>
#include <stdlib.h>
#include <gputimer.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void convolution(int width, int height) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < width) {
        // TODO - Implement convolution
    }
}

int main(int argc, char **argv) {
    char* input_png = argv[1];
    char* output_png = argv[2];
    int num_of_threads = atoi(argv[3]);

    int width = 0, height = 0;

    // TODO - Allocate memory for input and output images

    GpuTimer timer;
    timer.Start();

    // TODO - Copy input image to device

    convolution<<<1, num_of_threads>>>(width, height);

    cudaDeviceSynchronize();

    // TODO - Copy output image to host

    timer.Stop();
    printf("*** Time Elapsed: %g ms ***\n", timer.Elapsed());

    // TODO - Save output image

    // TODO - Free memory
    return 0;
}