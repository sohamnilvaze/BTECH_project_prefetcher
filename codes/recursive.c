#include <stdio.h>
#include <stdlib.h>

int recurse(int n) {
    if (n <= 0) return 0;
    return n + recurse(n - 1);  // Stack grows & shrinks
}

int main(int argc, char * argv[]) {
    int N = 1;
    if(argc > 1)
    {
        N = atoi(argv[1]);
    }
    int result = recurse(N);
    printf("Result=%d\n", result);
    return 0;
}
