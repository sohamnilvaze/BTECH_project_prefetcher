#include <stdio.h>
#define N 1024
int data[N];
int indexArr[N/2]; // indirect indices

int main() {
    for (int i = 0; i < N/2; i++) indexArr[i] = (i*3) % N;  // sparse pattern
    for (int i = 0; i < N/2; i++) {
        data[indexArr[i]] = i;  // indirect access
    }
    return 0;
}
