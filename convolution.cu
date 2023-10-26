#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lodepng.h"

__global__ void convolution(unsigned char *input, unsigned char *output, int width, int height) {
    int item_count = (width - 2) * (height - 2);

    int op_per_thread = item_count / (blockDim.x);
    if (item_count % (blockDim.x)) {
        op_per_thread++;
        
    }
    printf("op per thread: %d", op_per_thread);

    for(int op = 0; op < op_per_thread; op++) {
        int index = threadIdx.x + (blockDim.x) * op;

        if (index < item_count) {
        // Implement convolution
        printf("index: %d\n", index);
            int i = index / (width - 2);
            int j = index % (width - 2);

            // printf("%d %d\n", i, j);

            // for(int ii = 0; ii <= 2; ii++) {
            //     for(int jj = 0; jj <= 2; jj++) {

            //     }
            // }

        }   
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

    temp_output = (unsigned char *) malloc((width - 2) * (height - 2) * 4 * sizeof(unsigned char));

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

    printf("error: %s\n", cudaGetErrorString(cudaGetLastError()));

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