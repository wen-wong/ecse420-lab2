#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define P 0.5
#define ETA 0.0002
#define G 0.75
#define SIZE 512
#define SIMULATION_HIT 1;

// Apply the algorithm to the interior elements
__global__ void synthesis_interior_elements(float *u, float *u1, float *u2, int size, int num_of_elements) {
    // Calculate the number of operations per thread
    int op_per_thread = num_of_elements / (gridDim.x * blockDim.x);
    // If the number of operations is not divisible by the number of threads, add one more operation to the last thread
    if (num_of_elements % (gridDim.x * blockDim.x) != 0) {
        op_per_thread++;
    }

    // Iterate over the operations
    for (int op = 0; op < op_per_thread; op++) {
        // Calculate the index of the element to be processed by the current block and thread
        int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * op_per_thread + op;
        // If the index is out of bounds, return
        if (index >= size * size) {
            return;
        }

        // Calculate the row and column of the element
        int i = index / size;
        int j = index % size;

        // If the element is a boundary element or a corner element, skip it
        if (i == 0 || i == size - 1 || j == 0 || j == size - 1) {
            continue;
        }

        // Apply the algorithm to the element
        u[i * size + j] = 
            (P * (
                u1[(i - 1) * size + j] 
                + u1[(i + 1) * size + j] 
                + u1[i * size + (j - 1)] 
                + u1[i * size + (j + 1)] 
                - 4 * u1[i * size + j]) 
            + 2 * u1[i * size + j] 
            - (1 - ETA) * u2[i * size + j]
            ) / (1 + ETA);
    }

    return;
}

// Apply the algorithm to the boundary elements
__global__ void synthesis_boundary_elements(float *u, int size, int num_of_elements) {
    // Calculate the number of operations per thread
    int op_per_thread = num_of_elements / (gridDim.x * blockDim.x);
    // If the number of operations is not divisible by the number of threads, add one more operation to the last thread
    if (num_of_elements % (gridDim.x * blockDim.x) != 0) {
        op_per_thread++;
    }

    // Iterate over the operations
    for (int op = 0; op < op_per_thread; op++) {
        // Calculate the index of the element to be processed by the current block and thread
        int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * op_per_thread + op;
        // If the index is out of bounds, return
        if (index >= size * size) {
            return;
        }

        // Calculate the row and column of the element
        int i = index / size;
        int j = index % size;

        // Check if the element is not an interior or a corner element on the first row
        if (i == 0 && j != 0 && j != size - 1) {
            u[i * size + j] = G * u[1 * size + j];
        // Check if the element is not an interior or a corner element on the last row
        } else if (i == size - 1 && j != 0 && j != size - 1) {
            u[(size - 1) * size + j] = G * u[(size - 2) * size + j];
        // Check if the element is not an interior or a corner element on the first column
        } else if (i != 0 && i != size - 1 && j == 0) {
            u[i * size] = G * u[i * size + 1];
        // Check if the element is not an interior or a corner element on the last column
        } else if (i != 0 && i != size - 1 && j == size - 1) {
            u[i * size + (size - 1)] = G * u[i * size + (size - 2)];
        }
    }

    return;
}

// Apply the algorithm to the corner elements
__global__ void synthesis_corner_elements(float *u, int size, int num_of_elements) {
    // Calculate the number of operations per thread
    int op_per_thread = num_of_elements / (gridDim.x * blockDim.x);
    // If the number of operations is not divisible by the number of threads, add one more operation to the last thread
    if (num_of_elements % (gridDim.x * blockDim.x) != 0) {
        op_per_thread++;
    }

    // Iterate over the operations
    for (int op = 0; op < op_per_thread; op++) {
        // Calculate the index of the element to be processed by the current block and thread
        int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * op_per_thread + op;
        // If the index is out of bounds, return
        if (index >= size * size) {
            return;
        }

        // Calculate the row and column of the element
        int i = index / size;
        int j = index % size;

        // Check if the element is the top-left corner element
        if (i == 0 && j == 0) {
            u[0] = G * u[1 * size + 0];
        // Check if the element is the top-right corner element
        } else if (i == size - 1 && j == 0) {
            u[(size - 1) * size] = G * u[(size - 2) * size + 0];
        // Check if the element is the bottom-left corner element
        } else if (i == 0 && j == size - 1) {
            u[size - 1] = G * u[0 * size + (size - 2)];
        // Check if the element is the bottom-right corner element
        } else if (i == size - 1 && j == size - 1) {
            u[(size - 1) * size + (size - 1)] = G * u[(size - 1) * size + (size - 2)];
        }
    }

    return;
}

// Print the result of each iteration in position (size / 2, size / 2)
void print_result(float *result, int num_of_iterations) {
    for (int i = 0; i < num_of_iterations; i++) {
        printf("[%d]\t%f\n", i, result[i]);
    }

    return;
}

// Swap the two pointers
void swap(float **a, float **b) {
    float *temp = *a;
    *a = *b;
    *b = temp;

    return;
}

// Global synthesis function
void synthesis(float *u, float *u1, float *u2, int size, int num_of_iterations, float *result, int num_of_elements, int num_of_blocks, int num_of_threads) {
    float* u_temp = (float*) malloc(sizeof(float) * size * size);
    // Initialize hit on the center of the grid
    u_temp[(SIZE / 2 ) * SIZE + (SIZE / 2)] = SIMULATION_HIT;

    float* out = (float*) malloc(sizeof(float) * size * size);

    cudaMemcpy(u1, u_temp, sizeof(float) * size * size, cudaMemcpyHostToDevice);

    double elapsed = 0;


    for (int i = 0; i < num_of_iterations; i++) {
        GpuTimer timer;
        timer.Start();
        synthesis_interior_elements<<<num_of_blocks, num_of_threads>>>(u, u1, u2, size, num_of_elements);
        cudaDeviceSynchronize();
        synthesis_boundary_elements<<<num_of_blocks, num_of_threads>>>(u, size, num_of_elements);
        cudaDeviceSynchronize();
        synthesis_corner_elements<<<num_of_blocks, num_of_threads>>>(u, size, num_of_elements);
        cudaDeviceSynchronize();

        timer.Stop();
        elapsed += timer.Elapsed();

        cudaMemcpy(out, u, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
        
        result[i] = out[(SIZE / 2 ) * SIZE + (SIZE / 2)];

        swap(&u2, &u1);
        swap(&u1, &u);
    }

    print_result(result, num_of_iterations);

    printf("\n*** Time Elapsed: %f ms ***\n", elapsed);

    free(out);
    free(u_temp);

    return;
}

int main(int argc, char** argv) {
    int num_of_iterations = atoi(argv[1]);

    // Set the number of blocks and threads
    int num_of_blocks = 1024;
    int num_of_threads = 1024;
    
    // Allocate memory
    float *u, *u1, *u2, *result;
    cudaMalloc((void**) &u, sizeof(float) * SIZE * SIZE);
    cudaMalloc((void**) &u1, sizeof(float) * SIZE * SIZE);
    cudaMalloc((void**) &u2, sizeof(float) * SIZE * SIZE);
    result = (float *) calloc(num_of_iterations, sizeof(float));

    cudaMemset(u, 0, sizeof(float) * SIZE * SIZE);
    cudaMemset(u2, 0, sizeof(float) * SIZE * SIZE);

    synthesis(u, u1, u2, SIZE, num_of_iterations, result, SIZE * SIZE, num_of_blocks, num_of_threads);

    // Free memory
    cudaFree(u);
    cudaFree(u1);
    cudaFree(u2);
    free(result);

    return 0;
}
