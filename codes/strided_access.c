//Here the pattern type is stride length
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {
    int stride = 1; // default stride
    int N = 10;
    if (argc > 1) {
        stride = atoi(argv[1]);   // read stride from command line
        N = atoi(argv[2]);
    }
    int Nsq = N * N;
    // allocate an integer array
    int *arr = (int *)malloc(Nsq * sizeof(int));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // initialize array
    for (int i = 0; i < Nsq; i++) {
        arr[i] = i;
    }

    // perform strided access
    long long sum = 0;
    for (int i = 0; i < Nsq; i += stride) {
        sum += arr[i];   // read access
    }

    printf("Stride = %d,Nsq = %d, Sum = %lld\n", stride,Nsq,sum);

    free(arr);
    return 0;
}

