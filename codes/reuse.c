#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int N = 1024;
    int r = 2;
    if(argc > 1)
    {
        N = atoi(argv[1]);
        r = atoi(argv[2]);
    }
    int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i % r]++;  // repeatedly access only 8 elements
    }
    return 0;
}
