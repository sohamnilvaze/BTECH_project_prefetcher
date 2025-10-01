#include <stdio.h>
#define N 30
#define M 30

int main() {
    int A[N][M];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            A[i][j] = i + j;

    int sum = 0;
    for (int j = 0; j < M; j++)        // Column-first (strided)
        for (int i = 0; i < N; i++)
            sum += A[i][j];

    printf("Sum=%d\n", sum);
    return 0;
}
