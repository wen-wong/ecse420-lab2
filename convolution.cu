#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lodepng.h"
#include "wm.h"

__global__ void convolution(unsigned char *input, unsigned char *output, float *wm, int width, int height) {
    // Number of items to be processed
    int item_count = (width - 2) * (height - 2);

    // Number of operations per thread
    int op_per_thread = item_count / (blockDim.x);
    if (item_count % (blockDim.x)) {
        op_per_thread++;
    }

    for(int op = 0; op < op_per_thread; op++) {
        int index = threadIdx.x * op_per_thread + op;

        // Check if index is out of bound
        if (index >= item_count) {
            return;
        }

        int i = index / (width - 2);
        int j = index % (width - 2);

        // For each RGBA value
        for(int k = 0; k <= 3; k++) {

            float sum = 0;
            int integer_sum = 0;

            if (k == 3) {
                integer_sum = 255;
            } else {

                // Convolution
                for(unsigned long ii = 0; ii <= 2; ii++) {
                    for(unsigned long jj = 0; jj <= 2; jj++) {
                        sum += 1.0 * input[(i + ii) * width * 4 + (j + jj) * 4 + k] * wm[ii * 3 + jj] ;
                    }
                }

                // Rounding float to integer
                if (((int) (sum * 10)) % 10 >= 5) {
                    integer_sum = sum + 1;
                } else {
                    integer_sum = sum;
                }

                // Clamping integer to [0, 255]
                if (integer_sum < 0) {
                    integer_sum = 0;
                } else if (integer_sum > 255) {
                    integer_sum = 255;
                }
            }

            // Write to output
            output[index * 4 + k] = integer_sum;
        }
    }
}

int main(int argc, char *argv[]) {
    char* input_filename = argv[1];
    char* output_filename = argv[2];
    int num_of_threads = atoi(argv[3]);

    unsigned width, height;

    unsigned error;
    unsigned char *temp_image;
    unsigned char *temp_output;
    float *wm;
    unsigned char *input_image;
    unsigned char *output_image;

    // Load input image from file
    error = lodepng_decode32_file(&temp_image, &width, &height, input_filename);
    if (error) {
        printf("error %u: %s", error, lodepng_error_text(error));
    }

    temp_output = (unsigned char *) malloc((width - 2) * (height - 2) * 4 * sizeof(unsigned char));

    // Allocate memory for input and output images
    cudaMalloc((void **) &input_image, width * height * 4 * sizeof(unsigned char));
    cudaMalloc((void **) &output_image, (width - 2) * (height - 2) * 4 * sizeof(unsigned char));
    cudaMalloc((void **) &wm, 3 * 3 * sizeof(float));

    // Copy input image to device
    cudaMemcpy(input_image, temp_image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(wm, w, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.Start();

    convolution<<<1, num_of_threads>>>(input_image, output_image, wm, (int) width, (int) height);

    cudaDeviceSynchronize();

    timer.Stop();
    printf("*** Time Elapsed: %g ms ***\n", timer.Elapsed());

    printf("error: %s\n", cudaGetErrorString(cudaGetLastError()));

    // Copy output image to host
    cudaMemcpy(temp_output, output_image, (width - 2) * (height - 2) * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save output image
    lodepng_encode32_file(output_filename, temp_output, width - 2, height - 2);
    
    // Free memory
    free(temp_image);
    free(temp_output);
    cudaFree(input_image);
    cudaFree(output_image);
    
    return 0;
}