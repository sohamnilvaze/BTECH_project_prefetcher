#include <stdio.h>
#define N 64
#define B 8
int mat[N][N];

int main() {
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
