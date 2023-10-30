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
        u[0] = G * u[1 * size + 1];
    } else if (i == size - 1 && j == 0) {
        u[(size - 1) * size] = G * u[(size - 2) * size + 1];
    } else if (i == 0 && j == size - 1) {
        u[size - 1] = G * u[1 * size + (size - 2)];
    } else if (i == size - 1 && j == size - 1) {
        u[(size - 1) * size + (size - 1)] = G * u[(size - 2) * size + (size - 2)];
    }

    return;
}


void print_result(float *result, int num_of_iterations) {
    for (int i = 0; i < num_of_iterations; i++) {
        printf("[%d]\t%f\n", i, result[i]);
    }

    return;
}

void synthesis(float *u, float *d_u, float *d_u1, float *d_u2, int size, int num_of_iterations, float *result, int num_of_elements) {

    GpuTimer timer;
    timer.Start();

    for (int i = 0; i < num_of_iterations; i++) {
        synthesis_interior_elements<<<1, num_of_elements>>>(d_u, d_u1, d_u2, size);
        cudaDeviceSynchronize();
        synthesis_boundary_elements<<<1, num_of_elements>>>(d_u, size);
        cudaDeviceSynchronize();
        synthesis_corner_elements<<<1, num_of_elements>>>(d_u, size);
        cudaDeviceSynchronize();


        cudaMemcpy(d_u2, d_u1, sizeof(float) * size * size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_u1, d_u, sizeof(float) * size * size, cudaMemcpyDeviceToDevice);

        cudaMemcpy(u, d_u, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
        result[i] = u[2 * size + 2];
    }

    timer.Stop();
    double elapsed = timer.Elapsed();
    print_result(result, num_of_iterations);

    printf("*** Time Elapsed: %f ms ***\n", timer.Elapsed());

    return;
}

int main(int argc, char** argv) {
    int num_of_iterations = atoi(argv[1]);
    
    // Allocate memory
    float *u, *u1, *result;
    float *d_u, *d_u1, *d_u2;
    
    u = (float*) malloc(sizeof(float) * SIZE * SIZE);
    u1 = (float*) malloc(sizeof(float) * SIZE * SIZE);
    result = (float*) malloc(sizeof(float) * num_of_iterations);

    cudaMalloc((void**) &d_u, sizeof(float) * SIZE * SIZE);
    cudaMalloc((void**) &d_u1, sizeof(float) * SIZE * SIZE);
    cudaMalloc((void**) &d_u2, sizeof(float) * SIZE * SIZE);

    u1[(SIZE / 2) * SIZE + (SIZE / 2)] = SIMULATION_HIT;
    cudaMemcpy(d_u1, u1, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice);

    synthesis(u, d_u, d_u1, d_u2, SIZE, num_of_iterations, result, SIZE * SIZE);

    // Free memory
    free(u);
    free(u1);
    cudaFree(d_u);
    cudaFree(d_u1);
    cudaFree(d_u2);
    free(result);

    return 0;
}
