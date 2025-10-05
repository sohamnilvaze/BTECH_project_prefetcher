//Here the pattern type is stride length
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 5*5     // default array size
#endif

int main(int argc, char *argv[]) {
    int stride = 1; // default stride
    if (argc > 1) {
        stride = atoi(argv[1]);   // read stride from command line
    }

    // allocate an integer array
    int *arr = (int *)malloc(N * sizeof(int));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // initialize array
    for (int i = 0; i < N; i++) {
        arr[i] = i;
    }

    // perform strided access
    long long sum = 0;
    for (int i = 0; i < N; i += stride) {
        sum += arr[i];   // read access
    }

    printf("Stride = %d, Sum = %lld\n", stride, sum);

    free(arr);
    return 0;
}

