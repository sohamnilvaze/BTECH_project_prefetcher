#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
    int N = 10;
    if(argc > 1)
    {
        N = atoi(argv[1]);
    }
    double A[N];
    
    for (int i = 0; i < N; i++) A[i] = (double)i;

    double sum = 0.0;
    for (int i = 0; i < N; i++) {  // Sequential access
        sum += A[i];
    }
    printf("Sum calculated\n");
    return 0;
}
