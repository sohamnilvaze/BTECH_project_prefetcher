#include <stdio.h>

int recurse(int n) {
    if (n <= 0) return 0;
    return n + recurse(n - 1);  // Stack grows & shrinks
}

int main() {
    int result = recurse(100);
    printf("Result=%d\n", result);
    return 0;
}
