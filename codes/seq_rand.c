#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 1024
int arr[N];

int main() {
    srand(time(NULL));
    // Phase 1: sequential
    for (int i = 0; i < N; i++) arr[i] = i;
    // Phase 2: random
    for (int i = 0; i < N; i++) {
        int idx = rand() % N;
        arr[idx] = i;
    }
    return 0;
}
