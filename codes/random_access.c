#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 100

int main() {
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
