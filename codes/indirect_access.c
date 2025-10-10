#include <stdio.h>
#include <stdlib.h>



int main(int argc, char*argv[]) {
    int N = 1024;
    int stride = 3;
    if(argc > 1)
    {
        N = atoi(argv[1]);
        stride = atoi(argv[2]);
    }
    int data[N];
    int indexArr[N/2]; // indirect indices
    for (int i = 0; i < N/2; i++) indexArr[i] = (i*stride) % N;  // sparse pattern
    for (int i = 0; i < N/2; i++) {
        data[indexArr[i]] = i;  // indirect access
    }
    return 0;
}
