//Here the pattern type is block size

#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {
    int B = 1;
    int N = 128;
    int M = 128;
    if(argc > 1)
    {
        B = atoi(argv[1]);
        N = atoi(argv[2]);
        M = atoi(argv[3]);
    }
    int mat[N][M];
    for (int ii = 0; ii < N; ii += B) {
        for (int jj = 0; jj < M; jj += B) {
            for (int i = ii; i < ii + B; i++) {
                for (int j = jj; j < jj + B; j++) {
                    mat[i][j] = i + j;  // tile/block access
                }
            }
        }
    }
    return 0;
}
