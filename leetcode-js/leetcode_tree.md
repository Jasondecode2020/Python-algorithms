# Tree

## 1 Tree traversal

These are the tree traversal problem in leetcode:

1.1, 1.2 and 1.3 are nearly the same:
inorder, preorder, postorder

1.4, 1.5,1.6 and 1.7 are nearly the same:
level traversal top-bottom, bottom-top, zigzag, average

### 1.1 94. Binary Tree Inorder Traversal

Given the root of a binary tree, return the inorder traversal of its nodes' values.

Input: root = [1,null,2,3]
Output: [1,3,2]

```js
var inorderTraversal = function (root, res = []) {
  if (root) {
    inorderTraversal(root.left, res);
    res.push(root.val);
    inorderTraversal(root.right, res);
  }
  return res;
};
```

### 1.2 144. Binary Tree Preorder Traversal

Given the root of a binary tree, return the preorder traversal of its nodes' values.

Input: root = [1,null,2,3]
Output: [1,2,3]

```js
var preorderTraversal = function (root, res = []) {
  if (root) {
    res.push(root.val);
    preorderTraversal(root.left, res);
    preorderTraversal(root.right, res);
  }
  return res;
};
```

### 1.3 145. Binary Tree Postorder Traversal

Given the root of a binary tree, return the postorder traversal of its nodes' values.

Input: root = [1,null,2,3]
Output: [3,2,1]

```js
var postorderTraversal = function (root, res = []) {
  if (root) {
    postorderTraversal(root.left, res);
    postorderTraversal(root.right, res);
    res.push(root.val);
  }
  return res;
};
```

### 1.4 102. Binary Tree Level Order Traversal

Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

```js
var levelOrder = function (root) {
  if (!root) return [];

  let queue = [root];
  let result = [];
  while (queue.length > 0) {
    let level = [];
    let queueLength = queue.length;
    for (let i = 0; i < queueLength; i++) {
      let node = queue.shift();
      level.push(node.val);
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
    result.push(level);
  }
  return result;
};
```

### 1.5 107. Binary Tree Level Order Traversal II

Given the root of a binary tree, return the bottom-up level order traversal of its nodes' values. (i.e., from left to right, level by level from leaf to root).

```js
var levelOrderBottom = function (root) {
  if (!root) return [];

  let queue = [root];
  let result = [];
  while (queue.length > 0) {
    let level = [];
    let queueLength = queue.length;
    for (let i = 0; i < queueLength; i++) {
      let node = queue.shift();
      level.push(node.val);
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
    result.unshift(level);
  }
  return result;
};
```

### 1.6 103. Binary Tree Zigzag Level Order Traversal

Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]

```js
var zigzagLevelOrder = function (root) {
  if (!root) return [];

  let queue = [root];
  let result = [];
  let counter = 0;
  while (queue.length > 0) {
    counter++;
    let level = [];
    let queueLength = queue.length;
    for (let i = 0; i < queueLength; i++) {
      let node = queue.shift();
      level.push(node.val);
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
    counter % 2 ? result.push(level) : result.push(level.reverse());
  }
  return result;
};
```

### 1.7 637. Average of Levels in Binary Tree

Given the root of a binary tree, return the average value of the nodes on each level in the form of an array. Answers within 10-5 of the actual answer will be accepted.

Input: root = [3,9,20,null,null,15,7]
Output: [3.00000,14.50000,11.00000]
Explanation: The average value of nodes on level 0 is 3, on level 1 is 14.5, and on level 2 is 11.
Hence return [3, 14.5, 11].

```js
var averageOfLevels = function (root) {
  if (!root) return [];

  let queue = [root];
  let result = [];
  while (queue.length > 0) {
    let level = [];
    let queueLength = queue.length;
    for (let i = 0; i < queueLength; i++) {
      let node = queue.shift();
      level.push(node.val);
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
    let sum = level.reduce((accu, curr) => accu + curr, 0);
    result.push(sum / level.length);
  }
  return result;
};
```

### 1.8 1302. Deepest Leaves Sum

Given the root of a binary tree, return the sum of values of its deepest leaves.

Input: root = [1,2,3,4,5,null,6,7,null,null,null,null,8]
Output: 15

```js
var deepestLeavesSum = function (root) {
  if (!root) return [];
  let queue = [root];
  let result = [];
  while (queue.length > 0) {
    let level = [];
    let queueLength = queue.length;
    for (let i = 0; i < queueLength; i++) {
      let node = queue.shift();
      level.push(node.val);
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
    result.unshift(level);
  }
  return result[0].reduce((accu, curr) => accu + curr, 0);
};
```

### 1.9 590. N-ary Tree Postorder Traversal

Given the root of an n-ary tree, return the postorder traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)

Input: root = [1,null,3,2,4,null,5,6]
Output: [5,6,3,2,4,1]

```js
var postorder = function (root) {
  if (!root) return [];

  let res = [];
  if (root.children) {
    for (let child of root.children) {
      res.push(...postorder(child));
    }
  }
  res.push(root.val);
  return res;
};
```

### 1.10 589. N-ary Tree Preorder Traversal

Given the root of an n-ary tree, return the preorder traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)

Input: root = [1,null,3,2,4,null,5,6]
Output: [1,3,5,6,2,4]

```js
var preorder = function (root) {
  if (!root) return [];

  let res = [];
  res.push(root.val);
  if (root.children) {
    for (let child of root.children) {
      res.push(...preorder(child));
    }
  }
  return res;
};
```

### 1.11 404. Sum of Left Leaves

Given the root of a binary tree, return the sum of all left leaves.

Input: root = [3,9,20,null,null,15,7]
Output: 24
Explanation: There are two left leaves in the binary tree, with values 9 and 15 respectively.

```js
var sumOfLeftLeaves = function (root) {
  // BFS
  if (!root) return 0;
  if (root.left && !root.left.left && !root.left.right) {
    return root.left.val + sumOfLeftLeaves(root.right);
  }
  return sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
};
```

### 1.11 965. Univalued Binary Tree

A binary tree is uni-valued if every node in the tree has the same value.

Given the root of a binary tree, return true if the given tree is uni-valued, or false otherwise.

Input: root = [1,1,1,1,1,null,1]
Output: true

```js
var isUnivalTree = function (root, res = []) {
  if (root) {
    res.push(root.val);
    isUnivalTree(root.left, res);
    isUnivalTree(root.right, res);
  }
  return new Set(res).size === 1;
};
```

### 1.12 872. Leaf-Similar Trees

Consider all the leaves of a binary tree, from left to right order, the values of those leaves form a leaf value sequence.

For example, in the given tree above, the leaf value sequence is (6, 7, 4, 9, 8).

Two binary trees are considered leaf-similar if their leaf value sequence is the same.

Return true if and only if the two given trees with head nodes root1 and root2 are leaf-similar.

Input: root1 = [3,5,1,6,2,9,8,null,null,7,4], root2 = [3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]
Output: true

```js
var leafSimilar = function (root1, root2) {
  const dfs = (node) => {
    const stack = [node];
    let leaves = "";
    while (stack.length) {
      let curr = stack.pop();
      if (!curr.left && !curr.right) leaves += curr.val;
      if (curr.left) stack.push(curr.left);
      if (curr.right) stack.push(curr.right);
    }
    return leaves;
  };

  return dfs(root1) === dfs(root2);
};
```

### 1.13 129. Sum Root to Leaf Numbers

You are given the root of a binary tree containing digits from 0 to 9 only.

Each root-to-leaf path in the tree represents a number.

For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
Return the total sum of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a 32-bit integer.

A leaf node is a node with no children.

Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.

```js
var sumNumbers = function (root, num = 0) {
  if (!root) return 0;
  num = num * 10 + root.val;
  if (!root.left && !root.right) return num;
  return sumNumbers(root.left, num) + sumNumbers(root.right, num);
};
```

### 1.14 257. Binary Tree Paths

iven the root of a binary tree, return all root-to-leaf paths in any order.

A leaf is a node with no children.

Input: root = [1,2,3,null,5]
Output: ["1->2->5","1->3"]

```js
var binaryTreePaths = function (root, res = [], ans = []) {
  res += "->" + root.val;
  //remove the first -> that was attached to the root
  if (!root.left && !root.right) ans.push(res.substring(2));
  if (root.left) binaryTreePaths(root.left, res, ans);
  if (root.right) binaryTreePaths(root.right, res, ans);
  return ans;
};
```

### 1.15 116. Populating Next Right Pointers in Each Node

Medium

You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
int val;
Node *left;
Node *right;
Node \*next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

Example 1:

Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
Example 2:

Input: root = []
Output: []

Constraints:

The number of nodes in the tree is in the range [0, 212 - 1].
-1000 <= Node.val <= 1000

```js
var connect = function (root) {
  if (!root) return root;
  const queue = [];
  queue.push(root);
  while (queue.length > 0) {
    const size = queue.length;
    for (let i = 0; i < size; i++) {
      let node = queue.shift();
      if (i < size - 1) node.next = queue[0];
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
  }
  return root;
};
```

### 1.16 117. Populating Next Right Pointers in Each Node II

Given a binary tree

struct Node {
int val;
Node *left;
Node *right;
Node \*next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

Example 1:

Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]
Explanation: Given the above binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
Example 2:

Input: root = []
Output: []

Constraints:

The number of nodes in the tree is in the range [0, 6000].
-100 <= Node.val <= 100

Follow-up:

You may only use constant extra space.
The recursive approach is fine. You may assume implicit stack space does not count as extra space for this problem.

Same as 116

```js
var connect = function (root) {
  if (!root) return root;
  const queue = [];
  queue.push(root);
  while (queue.length > 0) {
    const size = queue.length;
    for (let i = 0; i < size; i++) {
      let node = queue.shift();
      if (i < size - 1) node.next = queue[0];
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
  }
  return root;
};
```

### 1.17 105. Construct Binary Tree from Preorder and Inorder Traversal

Medium

Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

Example 1:

Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
Example 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]

Constraints:

1 <= preorder.length <= 3000
inorder.length == preorder.length
-3000 <= preorder[i], inorder[i] <= 3000
preorder and inorder consist of unique values.
Each value of inorder also appears in preorder.
preorder is guaranteed to be the preorder traversal of the tree.
inorder is guaranteed to be the inorder traversal of the tree.

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder: return None # just need one
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1: mid + 1], inorder[: mid])
        root.right = self.buildTree(preorder[mid + 1: ], inorder[mid + 1:])
        return root
```

### 1.18 106. Construct Binary Tree from Inorder and Postorder Traversal

Medium

Given two integer arrays inorder and postorder where inorder is the inorder traversal of a binary tree and postorder is the postorder traversal of the same tree, construct and return the binary tree.

Example 1:

Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]
Example 2:

Input: inorder = [-1], postorder = [-1]
Output: [-1]

Constraints:

1 <= inorder.length <= 3000
postorder.length == inorder.length
-3000 <= inorder[i], postorder[i] <= 3000
inorder and postorder consist of unique values.
Each value of postorder also appears in inorder.
inorder is guaranteed to be the inorder traversal of the tree.
postorder is guaranteed to be the postorder traversal of the tree.

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder: return None
        root = TreeNode(postorder[-1])
        mid = inorder.index(postorder[-1])
        root.left = self.buildTree(inorder[:mid], postorder[:mid])
        root.right = self.buildTree(inorder[mid + 1: ], postorder[mid: -1])
        return root
```

### 1.19 513. Find Bottom Left Tree Value

Medium

Given the root of a binary tree, return the leftmost value in the last row of the tree.

Example 1:

Input: root = [2,1,3]
Output: 1
Example 2:

Input: root = [1,2,3,4,null,5,6,null,null,7]
Output: 7

Constraints:

The number of nodes in the tree is in the range [1, 104].
-231 <= Node.val <= 231 - 1

```js
var findBottomLeftValue = function (root) {
  if (!root) return [];

  let queue = [root];
  let result = [];
  while (queue.length > 0) {
    let level = [];
    let queueLength = queue.length;
    for (let i = 0; i < queueLength; i++) {
      let node = queue.shift();
      level.push(node.val);
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
    result.push(level);
  }
  return result.pop()[0];
};
```

### 1.20 515. Find Largest Value in Each Tree Row

Medium

Given the root of a binary tree, return an array of the largest value in each row of the tree (0-indexed).

Example 1:

Input: root = [1,3,2,5,3,null,9]
Output: [1,3,9]
Example 2:

Input: root = [1,2,3]
Output: [1,3]
Example 3:

Input: root = [1]
Output: [1]
Example 4:

Input: root = [1,null,2]
Output: [1,2]
Example 5:

Input: root = []
Output: []

Constraints:

The number of nodes in the tree will be in the range [0, 104].
-231 <= Node.val <= 231 - 1

```js
var largestValues = function (root) {
  if (!root) return [];

  let queue = [root];
  let result = [];
  while (queue.length > 0) {
    let level = [];
    let queueLength = queue.length;
    for (let i = 0; i < queueLength; i++) {
      let node = queue.shift();
      level.push(node.val);
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
    result.push(level);
  }
  let ans = [];
  for (let item of result) {
    ans.push(Math.max(...item));
  }
  return ans;
};
```
