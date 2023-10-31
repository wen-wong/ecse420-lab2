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

__global__ void synthesis_interior_elements(float *u, float *u1, float *u2, int size, int num_of_elements) {
    int op_per_thread = num_of_elements / (gridDim.x * blockDim.x);
    if (num_of_elements % (gridDim.x * blockDim.x) != 0) {
        op_per_thread++;
    }

    for (int op = 0; op < op_per_thread; op++) {
        int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * op_per_thread + op;
        if (index >= size * size) {
            return;
        }

        int i = index / size;
        int j = index % size;

        if (i == 0 || i == size - 1 || j == 0 || j == size - 1) {
            break;       // TODO: not sure if we need  to change this to continue
        }

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

__global__ void synthesis_boundary_elements(float *u, int size, int num_of_elements) {
    int op_per_thread = num_of_elements / (gridDim.x * blockDim.x);
    if (num_of_elements % (gridDim.x * blockDim.x) != 0) {
        op_per_thread++;
    }

    for (int op = 0; op < op_per_thread; op++) {
        int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * op_per_thread + op;
        if (index >= size * size) {
            return;
        }

        int i = index / size;
        int j = index % size;

        if (i == 0 && j != 0 && j != size - 1) {
            u[i * size + j] = G * u[1 * size + j];
        } else if (i == size - 1 && j != 0 && j != size - 1) {
            u[(size - 1) * size + j] = G * u[(size - 2) * size + j];
        } else if (i != 0 && i != size - 1 && j == 0) {
            u[i * size] = G * u[i * size + 1];
        } else if (i != 0 && i != size - 1 && j == size - 1) {
            u[i * size + (size - 1)] = G * u[i * size + (size - 2)];
        }
    }

    return;
}

__global__ void synthesis_corner_elements(float *u, int size, int num_of_elements) {
    int op_per_thread = num_of_elements / (gridDim.x * blockDim.x);
    if (num_of_elements % (gridDim.x * blockDim.x) != 0) {
        op_per_thread++;
    }

    for (int op = 0; op < op_per_thread; op++) {
        int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * op_per_thread + op;
        if (index >= size * size) {
            return;
        }

        int i = index / size;
        int j = index % size;

        if (i == 0 && j == 0) {
            u[0] = G * u[1 * size + 0];
        } else if (i == size - 1 && j == 0) {
            u[(size - 1) * size] = G * u[(size - 2) * size + 0];
        } else if (i == 0 && j == size - 1) {
            u[size - 1] = G * u[0 * size + (size - 2)];
        } else if (i == size - 1 && j == size - 1) {
            u[(size - 1) * size + (size - 1)] = G * u[(size - 1) * size + (size - 2)];
        }
    }

    return;
}

void print_result(float *result, int num_of_iterations) {
    for (int i = 0; i < num_of_iterations; i++) {
        printf("[%d]\t%f\n", i, result[i]);
    }

    return;
}

void swap(float **a, float **b) {
    float *temp = *a;
    *a = *b;
    *b = temp;

    return;
}

void synthesis(float *u, float *u1, float *u2, int size, int num_of_iterations, float *result, int num_of_elements, int num_of_blocks, int num_of_threads) {
    u1[(SIZE / 2 ) * SIZE + (SIZE / 2)] = 1;

    GpuTimer timer;
    timer.Start();

    for (int i = 0; i < num_of_iterations; i++) {
        synthesis_interior_elements<<<num_of_blocks, num_of_threads>>>(u, u1, u2, size, num_of_elements);
        cudaDeviceSynchronize();
        synthesis_boundary_elements<<<num_of_blocks, num_of_threads>>>(u, size, num_of_elements);
        cudaDeviceSynchronize();
        synthesis_corner_elements<<<num_of_blocks, num_of_threads>>>(u, size, num_of_elements);
        cudaDeviceSynchronize();

        result[i] = u[(size / 2) * size + (size / 2)];

        swap(&u2, &u1);
        swap(&u1, &u);
    }

    timer.Stop();
    double elapsed = timer.Elapsed();
    print_result(result, num_of_iterations);

    printf("*** Time Elapsed: %f ms ***\n", elapsed);

    return;
}

int main(int argc, char** argv) {
    int num_of_iterations = atoi(argv[1]);

    int num_of_blocks = 16;
    int num_of_threads = 1024;
    
    // Allocate memory
    float *u, *u1, *u2, *result;
    cudaMallocManaged((void**) &u, sizeof(float) * SIZE * SIZE);
    cudaMallocManaged((void**) &u1, sizeof(float) * SIZE * SIZE);
    cudaMallocManaged((void**) &u2, sizeof(float) * SIZE * SIZE);
    cudaMallocManaged((void**) &result, sizeof(float) * num_of_iterations);

    synthesis(u, u1, u2, SIZE, num_of_iterations, result, SIZE * SIZE, num_of_blocks, num_of_threads);

    // Free memory
    cudaFree(u);
    cudaFree(u1);
    cudaFree(u2);
    cudaFree(result);

    return 0;
}
