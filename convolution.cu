#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lodepng.h"
#include "wm.h"

__global__ void convolution(unsigned char *input, unsigned char *output, float *wm, int width, int height) {
    int item_count = (width - 2) * (height - 2);
    int op_per_thread = item_count / (blockDim.x);
    if (item_count % (blockDim.x)) {
        op_per_thread++;
    }

    for(int op = 0; op < op_per_thread; op++) {
        int index = threadIdx.x * op_per_thread + op;
        // if (threadIdx.x == 0) {
        //     printf("op: %d, index: %d\n", op, index);
        // } else {
        //     break;
        // }

        if (index >= item_count) {
            return;
        }
        // printf("index: %d\n", index);
        int i = index / (width - 2);
        int j = index % (width - 2);

        if (!(i >= 1 && i <= (height) - 1) || !(j >= 1 && j <= (width) - 1)) {
            continue;
        }

        float r = 0, g = 0, b = 0, a = 0;

        for(int ii = 0; ii <= 2; ii++) {
            for(int jj = 0; jj <= 2; jj++) {
                // printf("i: %d, j: %d, ii: %d, jj: %d\n", i, j, ii, jj);
                for(int k = 0; k <= 3; k++) {
                    // output[index * 4 + k] += input[(i + ii - 1) * width * 4 + (j + jj - 1) * 4 + k] * wm[ii * 3 + jj];
                    // if (k == 3) {
                    //     output[index * 4 + k] = 255;
                    // } else {
                    //     output[index * 4 + k] += input[index * 4 + k];
                    // }
                    // output[index * 4 + k] 
                    r += input[(i + ii - 1) * width * 4 + (j + jj - 1) * 4] * wm[ii * 3 + jj];
                    g += input[(i + ii - 1) * width * 4 + (j + jj - 1) * 4 + 1] * wm[ii * 3 + jj];
                    b += input[(i + ii - 1) * width * 4 + (j + jj - 1) * 4 + 2] * wm[ii * 3 + jj];
                    a += input[(i + ii - 1) * width * 4 + (j + jj - 1) * 4 + 3] * wm[ii * 3 + jj];
                    // if (index == 10000) {
                    //     printf("[%d][%d] k: %d\n", (i + ii - 1) + (i * width), (j + jj - 1), k);
                    // }
                }
            }
        }

        output[index * 4] = r;
        output[index * 4 + 1] = g;
        output[index * 4 + 2] = b;
        output[index * 4 + 3] = a;
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