#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int value;
    struct Node* next;
} Node;

// Function to shuffle an array (Fisher-Yates shuffle)
void shuffle(Node** array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Node* temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

int main(int argc, char* argv[]) {
    int N = 1024;  // default linked list length
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    srand(time(NULL)); // random seed for pointer shuffling

    // Step 1: Allocate all nodes
    Node** nodes = malloc(N * sizeof(Node*));
    for (int i = 0; i < N; i++) {
        nodes[i] = malloc(sizeof(Node));
        nodes[i]->value = i;
        nodes[i]->next = NULL;
    }

    // Step 2: Randomly shuffle node order
    shuffle(nodes, N);

    // Step 3: Link nodes in shuffled order
    for (int i = 0; i < N - 1; i++) {
        nodes[i]->next = nodes[i + 1];
    }
    nodes[N - 1]->next = NULL; // last node terminates list

    // Step 4: Traverse the randomized linked list
    Node* curr = nodes[0];
    long long sum = 0;
    while (curr != NULL) {
        sum += curr->value;
        curr = curr->next;
    }

    printf("Traversal complete. Sum = %lld\n", sum);

    // Step 5: Cleanup (optional for small benchmarks)
    for (int i = 0; i < N; i++) {
        free(nodes[i]);
    }
    free(nodes);

    return 0;
}
