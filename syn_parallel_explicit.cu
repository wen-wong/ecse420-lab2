#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define P 0.5
#define ETA 0.0002
#define G 0.75
#define SIZE 4
#define SIMULATION_HIT 1;

__global__ void synthesis_interior_elements(float *u, float *u1, float *u2, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size * size) {
        return;
    }

    int i = index / size;
    int j = index % size;

    if (i == 0 || i == size - 1 || j == 0 || j == size - 1) {
        return;
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

    return;
}

__global__ void synthesis_boundary_elements(float *u, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

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

    return;
}

__global__ void synthesis_corner_elements(float *u, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

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

void synthesis(float *u, float *u1, float *u2, int size, int num_of_iterations, float *result, int num_of_elements) {
    float* u_temp = (float*) malloc(sizeof(float) * size * size);
    u_temp[(SIZE / 2 ) * SIZE + (SIZE / 2)] = SIMULATION_HIT;

    float* out = (float*) malloc(sizeof(float) * size * size);

    cudaMemcpy(u1, u_temp, sizeof(float) * size * size, cudaMemcpyHostToDevice);

    double elapsed = 0.0;

    for (int i = 0; i < num_of_iterations; i++) {
        GpuTimer timer;
        timer.Start();
        synthesis_interior_elements<<<1, num_of_elements>>>(u, u1, u2, size);
        cudaDeviceSynchronize();
        synthesis_boundary_elements<<<1, num_of_elements>>>(u, size);
        cudaDeviceSynchronize();
        synthesis_corner_elements<<<1, num_of_elements>>>(u, size);
        cudaDeviceSynchronize();
        timer.Stop();
        elapsed += timer.Elapsed();

        cudaMemcpy(out, u, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
        
        result[i] = out[(SIZE / 2 ) * SIZE + (SIZE / 2)];

        swap(&u2, &u1);
        swap(&u1, &u);
    }

    print_result(result, num_of_iterations);

    printf("*** Time Elapsed: %f ms ***\n", elapsed);

    return;
}

int main(int argc, char** argv) {
    int num_of_iterations = atoi(argv[1]);
    
    // Allocate memory
    float *u, *u1, *u2, *result;
    cudaMalloc((void**) &u, sizeof(float) * SIZE * SIZE);
    cudaMalloc((void**) &u1, sizeof(float) * SIZE * SIZE);
    cudaMalloc((void**) &u2, sizeof(float) * SIZE * SIZE);
    cudaMalloc((void**) &result, sizeof(float) * num_of_iterations);

    cudaMemset(u, 0, sizeof(float) * SIZE * SIZE);
    cudaMemset(u1, 0, sizeof(float) * SIZE * SIZE);
    cudaMemset(u2, 0, sizeof(float) * SIZE * SIZE);
    cudaMemset(result, 0, sizeof(float) * SIZE * SIZE);

    synthesis(u, u1, u2, SIZE, num_of_iterations, result, SIZE * SIZE);

    // Free memory
    cudaFree(u);
    cudaFree(u1);
    cudaFree(u2);
    cudaFree(result);

    return 0;
}
