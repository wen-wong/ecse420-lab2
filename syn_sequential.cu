#include <stdio.h>
#include <stdlib.h>
#include "cputimer.h"

#define P 0.5
#define ETA 0.0002
#define G 0.75
#define SIZE 4
#define SIMULATION_HIT 1;

void synthesis_interior_elements(float *u, float *u1, float *u2, int size) {
    for (int i = 1; i <= size - 2; i++) {
        for (int j = 1; j <= size - 2; j++) {
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
    }

    return;
}

void synthesis_boundary_elements(float *u, int size) {
    for (int i = 1; i <= size - 2; i++) {
        u[i] = G * u[1 * size + i];
        u[(size - 1) * size + i] = G * u[(size - 2) * size + i];
        u[i * size] = G * u[i * size + 1];
        u[i * size + (size - 1)] = G * u[i * size + (size - 2)];
    }

    return;
}

void synthesis_corner_elements(float *u, int size) {
    u[0] = G * u[1 * size + 1];
    u[(size - 1) * size] = G * u[(size - 2) * size + 1];
    u[size - 1] = G * u[1 * size + (size - 2)];
    u[(size - 1) * size + (size - 1)] = G * u[(size - 2) * size + (size - 2)];

    return;
}

void synthesis(float * u, float *u1, float *u2, int size, int num_of_iterations, float *result) {
    u1[(size / 2 ) * size + (size / 2)] = SIMULATION_HIT;

    for (int i = 0; i < num_of_iterations; i++) {
        synthesis_interior_elements(u, u1, u2, size);
        synthesis_boundary_elements(u, size);
        synthesis_corner_elements(u, size);

        memcpy(u2, u1, sizeof(float) * size * size);
        memcpy(u1, u, sizeof(float) * size * size);

        result[i] = u[2 * size + 2];
    }

    return;
}

void print_result(float *result, int num_of_iterations) {
    for (int i = 0; i < num_of_iterations; i++) {
        printf("[%d]\t%f\n", i, result[i]);
    }

    return;
}

int main(int argc, char** argv) {
    int num_of_iterations = atoi(argv[1]);

    // Allocate memory
    float *u = (float*) malloc(sizeof(float) * SIZE * SIZE);
    float *u1 = (float*) malloc(sizeof(float) * SIZE * SIZE);
    float *u2 = (float*) malloc(sizeof(float) * SIZE * SIZE);

    float *result = (float*) malloc(sizeof(float) * num_of_iterations);

    CpuTimer timer;
    timer.Start();

    // Run the algorithm
    synthesis(u, u1, u2, SIZE, num_of_iterations, result);

    timer.Stop();
    double elapsed = timer.Elapsed();

    print_result(result, num_of_iterations);

    printf("\n*** Time Elapsed: %f ms ***\n", elapsed);

    // Free memory
    free(u);
    free(u1);
    free(u2);
    free(result);

    return 0;
}
