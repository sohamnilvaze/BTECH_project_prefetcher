#include <stdio.h>
#define N 30
#define M 30

int main() {
    int A[N][M];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            A[i][j] = i + j;

    int sum = 0;
    for (int i = 0; i < N; i++)        // Row-first (cache friendly)
        for (int j = 0; j < M; j++)
            sum += A[i][j];

    printf("Sum=%d\n", sum);
    return 0;
}
