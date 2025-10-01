#include <stdio.h>
#define N 100

int main() {
    int A[N];
    for (int i = 0; i < N; i++) A[i] = i;

    int sum = 0;
    for (int i = 0; i < N; i++) {  // Sequential access
        sum += A[i];
    }
    printf("Sum=%d\n", sum);
    return 0;
}
