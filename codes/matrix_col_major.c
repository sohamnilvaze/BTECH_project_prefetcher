#include <stdio.h>
#include <stdlib.h>

int main(int argc, char*argv[]) {
    int N = 30;
    int M = 30;
    if(argc > 1)
    {
        N = atoi(argv[1]);
        M = atoi(argv[2]);
    }
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
