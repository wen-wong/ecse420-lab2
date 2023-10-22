#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "cputimer.h"

void synthesis() {
    // TODO - implement synthesis
}

int main(int argc, char** argv) {
    int num_of_iterations = atoi(argv[1]);

    CpuTimer timer;
    timer.Start();

    for (int i = 0; i < num_of_iterations; i++) {
        synthesis();
    }

    timer.Stop();
    printf("*** Time Elapsed: %f ms ***\n", timer.Elapsed());

    // TODO - Free memory

    return 0;
}
