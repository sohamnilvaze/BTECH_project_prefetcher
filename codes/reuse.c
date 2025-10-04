#include <stdio.h>
#define N 1024
int arr[N];

int main() {
    for (int i = 0; i < 100000; i++) {
        arr[i % 8]++;  // repeatedly access only 8 elements
    }
    return 0;
}
