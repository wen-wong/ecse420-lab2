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
    u[0] = G * u[1 * size + 0];
    u[(size - 1) * size] = G * u[(size - 2) * size + 0];
    u[size - 1] = G * u[0 * size + (size - 2)];
    u[(size - 1) * size + (size - 1)] = G * u[(size - 1) * size + (size - 2)];

    return;
}

void swap(float **a, float **b) {
    float *temp = *a;
    *a = *b;
    *b = temp;

    return;
}

void print_result(float *result, int num_of_iterations) {
    for (int i = 0; i < num_of_iterations; i++) {
        printf("[%d]\t%f\n", i, result[i]);
    }

    return;
}

void synthesis(float * u, float *u1, float *u2, int size, int num_of_iterations, float *result) {
    u1[(size / 2 ) * size + (size / 2)] = SIMULATION_HIT;

    double elapsed = 0;

    for (int i = 0; i < num_of_iterations; i++) {
        CpuTimer timer;
        timer.Start();
        
        synthesis_interior_elements(u, u1, u2, size);
        
        synthesis_boundary_elements(u, size);
        
        synthesis_corner_elements(u, size);

        timer.Stop();
        elapsed += timer.Elapsed();

        result[i] = u[(size / 2 ) * size + (size / 2)];
        
        swap(&u2, &u1);
        swap(&u1, &u);
    }

    print_result(result, num_of_iterations);

    printf("\n*** Time Elapsed: %f ms ***\n", elapsed);

    return;
}

int main(int argc, char** argv) {
    int num_of_iterations = atoi(argv[1]);

    // Allocate memory
    float *u = (float*) calloc(SIZE * SIZE, sizeof(float));
    float *u1 = (float*) calloc(SIZE * SIZE, sizeof(float));
    float *u2 = (float*) calloc(SIZE * SIZE, sizeof(float));

    float *result = (float*) calloc(num_of_iterations, sizeof(float));

    // Run the algorithm
    synthesis(u, u1, u2, SIZE, num_of_iterations, result);

    // Free memory
    free(u);
    free(u1);
    free(u2);
    free(result);

    return 0;
}
