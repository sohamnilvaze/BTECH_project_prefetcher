//Here the pattern type is block size

#include <stdio.h>
#define N 128
int mat[N][N];

int main(int argc, char *argv[]) {
    int B = 1;
    if(argc > 1)
    {
        B = atoi(argv[1]);
    }
    for (int ii = 0; ii < N; ii += B) {
        for (int jj = 0; jj < N; jj += B) {
            for (int i = ii; i < ii + B; i++) {
                for (int j = jj; j < jj + B; j++) {
                    mat[i][j] = i + j;  // tile/block access
                }
            }
        }
    }
    return 0;
}
