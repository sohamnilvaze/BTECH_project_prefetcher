#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
    int N = 10;
    if(argc > 1)
    {
        N = atoi(argv[1]);
    }
    int A[N];
    
    for (int i = 0; i < N; i++) A[i] = i;

    int sum = 0;
    for (int i = 0; i < N; i++) {  // Sequential access
        sum += A[i];
    }
    printf("Sum=%d\n", sum);
    return 0;
}
