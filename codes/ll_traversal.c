#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int val;
    struct Node* next;
} Node;

int main(int argc, char*argv[]) {
    int N = 1;
    if(argc > 1)
    {
        N = atoi(argv[1]);
    }
    Node* head = NULL;
    Node* prev = NULL;
    for (int i = 0; i < N; i++) {
        Node* node = malloc(sizeof(Node));
        node->val = i;
        node->next = NULL;
        if (prev) prev->next = node;
        else head = node;
        prev = node;
    }

    int sum = 0;
    Node* curr = head;
    while (curr) {
        sum += curr->val;  // Pointer chasing
        curr = curr->next;
    }
    printf("Sum=%d\n", sum);
    return 0;
}
