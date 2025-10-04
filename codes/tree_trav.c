#include <stdio.h>
#include <stdlib.h>
#define N 15

typedef struct Node {
    int val;
    struct Node* left;
    struct Node* right;
} Node;

Node* buildTree(int depth) {
    if (depth == 0) return NULL;
    Node* node = (Node*)malloc(sizeof(Node));
    node->val = depth;
    node->left = buildTree(depth - 1);
    node->right = buildTree(depth - 1);
    return node;
}

void dfs(Node* root) {
    if (!root) return;
    root->val += 1;  // access node
    dfs(root->left);
    dfs(root->right);
}

int main() {
    Node* root = buildTree(4);
    dfs(root);
    return 0;
}
