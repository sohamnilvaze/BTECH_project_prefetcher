#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char*argv[]) {
    int N = 1;
    if(argc > 1)
    {
        N = atoi(argv[1]);
    }
    int A[N];
    srand(time(NULL));
    for (int i = 0; i < N; i++) A[i] = i;

    int sum = 0;
    for (int i = 0; i < N; i++) {
        int idx = rand() % N;  // Random index
        sum += A[idx];
    }
    printf("Sum=%d\n", sum);
    return 0;
}
