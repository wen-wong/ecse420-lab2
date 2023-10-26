#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lodepng.h"

__global__ void convolution(unsigned char *input, unsigned char *output, int width, int height) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < width) {
        // TODO - Implement convolution
    }
}

int main(int argc, char **argv) {
    char* input_png = argv[1];
    char* output_png = argv[2];
    int num_of_threads = atoi(argv[3]);

    unsigned width, height;


    unsigned error;
    unsigned char *temp_image;
    unsigned char *temp_output;
    unsigned char *input_image;
    unsigned char *output_image;

    error = lodepng_decode32_file(&temp_image, &width, &height, input_png);
    if (error) {
        printf("error %u: %s", error, lodepng_error_text(error));
    }


    // Allocate memory for input and output images
    cudaMalloc((void **) &input_image, width * height * 4 * sizeof(unsigned char));
    cudaMalloc((void **) &output_image, (width - 2) * (height - 2) * 4 * sizeof(unsigned char));


    // Copy input image to device
    cudaMemcpy(input_image, temp_image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.Start();

    convolution<<<1, num_of_threads>>>(input_image, output_image, width, height);

    cudaDeviceSynchronize();

    timer.Stop();
    printf("*** Time Elapsed: %g ms ***\n", timer.Elapsed());

    // Copy output image to host
    cudaMemcpy(temp_output, output_image, (width - 2) * (height - 2) * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save output image
    lodepng_encode32_file(output_png, temp_output, width - 2, height - 2);
    
    // Free memory
    free(temp_image);
    free(temp_output);
    cudaFree(input_image);
    cudaFree(output_image);
    
    return 0;
}