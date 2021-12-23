# problems 62

easy: 14
medium: 43
hard: 5

## 1 1290. Convert Binary Number in a Linked List to Integer

Easy: 2 min

Given head which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.

Example 1:

Input: head = [1,0,1]
Output: 5
Explanation: (101) in base 2 = (5) in base 10
Example 2:

Input: head = [0]
Output: 0

Constraints:

The Linked List is not empty.
Number of nodes will not exceed 30.
Each node's value is either 0 or 1.

### method 1: stack

time: n, space: n

```js
var getDecimalValue = function (head) {
  let stack = [];
  while (head) {
    stack.push(head.val);
    head = head.next;
  }

  let res = 0;
  let i = 0;
  while (stack.length > 0) {
    let num = stack.pop();
    res += num * Math.pow(2, i);
    i += 1;
  }
  return res;
};
```

### method 2: bit operation

time: n, space: n

```js
var getDecimalValue = function (head) {
  let res = 0;
  while (head) {
    res = 2 * res + head.val;
    head = head.next;
  }
  return res;
};
```

## 2 1474. Delete N Nodes After M Nodes of a Linked List

Delete N Nodes After M Nodes of a Linked List
Given the head of a linked list and two integers m and n. Traverse the linked list and remove some nodes in the following way:
Start with the head as the current node.
Keep the first m nodes starting with the current node.
Remove the next n nodes
Keep repeating steps 2 and 3 until you reach the end of the list.
Return the head of the modified list after removing the mentioned nodes.
Follow up question: How can you solve this problem by modifying the list in-place?
Example 1:

Input: head = [1,2,3,4,5,6,7,8,9,10,11,12,13], m = 2, n = 3
Output: [1,2,6,7,11,12]
Explanation: Keep the first (m = 2) nodes starting from the head of the linked List (1 ->2) show in black nodes.
Delete the next (n = 3) nodes (3 -> 4 -> 5) show in read nodes.
Continue with the same procedure until reaching the tail of the Linked List.
Head of linked list after removing nodes is returned.

Input: head = [1,2,3,4,5,6,7,8,9,10,11], m = 1, n = 3
Output: [1,5,9]
Explanation: Head of linked list after removing nodes is returned.

Constraints:
The given linked list will contain between 1 and 10^4 nodes.
The value of each node in the linked list will be in the range [1, 10^6].
1 <= m,n <= 1000

### 1 2095. Delete the Middle Node of a Linked List

Medium

You are given the head of a linked list. Delete the middle node, and return the head of the modified linked list.

The middle node of a linked list of size n is the ⌊n / 2⌋th node from the start using 0-based indexing, where ⌊x⌋ denotes the largest integer less than or equal to x.

For n = 1, 2, 3, 4, and 5, the middle nodes are 0, 1, 1, 2, and 2, respectively.

Example 1:

Input: head = [1,3,4,7,1,2,6]
Output: [1,3,4,1,2,6]
Explanation:
The above figure represents the given linked list. The indices of the nodes are written below.
Since n = 7, node 3 with value 7 is the middle node, which is marked in red.
We return the new list after removing this node.
Example 2:

Input: head = [1,2,3,4]
Output: [1,2,4]
Explanation:
The above figure represents the given linked list.
For n = 4, node 2 with value 3 is the middle node, which is marked in red.

method: 1

```js
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
var deleteMiddle = function (head) {
  let dummy = new ListNode(0);
  dummy.next = head;
  let slow = dummy;
  let fast = dummy;
  let pre_slow = dummy;
  let temp = dummy;
  while (fast && fast.next) {
    fast = fast.next.next;
    slow = slow.next;
  }
  if (fast) {
    slow.next = slow.next.next;
  } else {
    while (pre_slow.next !== slow) {
      pre_slow = pre_slow.next;
    }
    pre_slow.next = slow.next;
  }

  return dummy.next;
};
```

The code is not easy to read, there are 2 while loop
The temp is not used, need to delete.

Now, I simplified the code to 2nd version

method: 2

```js
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
var deleteMiddle = function (head) {
  let dummy = new ListNode(0);
  dummy.next = head;
  let slow = head;
  let fast = head;
  let pre_slow = dummy;
  while (fast && fast.next) {
    fast = fast.next.next;
    slow = slow.next;
    pre_slow = pre_slow.next;
  }
  pre_slow.next = slow.next;

  return dummy.next;
};
```

However, it can be easier, just use 2 pointers

method: 3

```js
var deleteMiddle = function (head) {
  if (!head.next) return null;
  let slow = head;
  let fast = head.next.next;
  while (fast && fast.next) {
    fast = fast.next.next;
    slow = slow.next;
  }
  slow.next = slow.next.next;

  return head;
};
```

### 2 2058. Find the Minimum and Maximum Number of Nodes Between Critical Points

Medium

A critical point in a linked list is defined as either a local maxima or a local minima.

A node is a local maxima if the current node has a value strictly greater than the previous node and the next node.

A node is a local minima if the current node has a value strictly smaller than the previous node and the next node.

Note that a node can only be a local maxima/minima if there exists both a previous node and a next node.

Given a linked list head, return an array of length 2 containing [minDistance, maxDistance] where minDistance is the minimum distance between any two distinct critical points and maxDistance is the maximum distance between any two distinct critical points. If there are fewer than two critical points, return [-1, -1].

Example 1:

Input: head = [3,1]
Output: [-1,-1]
Explanation: There are no critical points in [3,1].

Example 2:

Input: head = [5,3,1,2,5,1,2]
Output: [1,3]
Explanation: There are three critical points:

- [5,3,1,2,5,1,2]: The third node is a local minima because 1 is less than 3 and 2.
- [5,3,1,2,5,1,2]: The fifth node is a local maxima because 5 is greater than 2 and 1.
- [5,3,1,2,5,1,2]: The sixth node is a local minima because 1 is less than 5 and 2.
  The minimum distance is between the fifth and the sixth node. minDistance = 6 - 5 = 1.
  The maximum distance is between the third and the sixth node. maxDistance = 6 - 3 = 3.

```js
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @return {number[]}
 */
var nodesBetweenCriticalPoints = function (head) {
  // all points pointed to index, if not find, set null
  min_point = max_point = last_point = null;
  // if set max val 100000, it will be faster, here just set value MAX_SAFE_INTEGER
  min_dist = Number.MAX_SAFE_INTEGER;
  // critical val before current critical val
  prev_val = head.val;
  // at least 2 val
  head = head.next;
  // start check from index 1
  idx = 1;
  while (head.next) {
    if (
      (head.val > prev_val && head.val > head.next.val) ||
      (head.val < prev_val && head.val < head.next.val)
    ) {
      if (min_point === null) {
        min_point = idx; // set first min_point
      } else {
        max_point = idx; // max_point is updated if there is
      }
      if (last_point) {
        // if there is last_point, check min_dist
        min_dist = Math.min(min_dist, idx - last_point);
      }
      last_point = idx; // update last_point
    }
    // normal loop to next index
    prev_val = head.val;
    idx++;
    head = head.next;
  }
  // check if min_dist is not changed, means no critical points
  if (min_dist === Number.MAX_SAFE_INTEGER) {
    min_dist = -1;
  }
  // check if there is max_point updated
  const max_dist = max_point ? max_point - min_point : -1;

  return [min_dist, max_dist];
};
```

more clear
method: 2

```js
var nodesBetweenCriticalPoints = function (head) {
  const isCriticalPoint = (prev, curr, next) =>
    (curr > prev_val && curr > next) || (curr < prev_val && curr < next);
  min_point = max_point = last_point = null;
  min_dist = Number.MAX_SAFE_INTEGER;
  prev_val = head.val;
  head = head.next;
  idx = 1;
  while (head.next) {
    if (isCriticalPoint(prev_val, head.val, head.next.val)) {
      !min_point ? (min_point = idx) : (max_point = idx);
      if (last_point) min_dist = Math.min(min_dist, idx - last_point);
      last_point = idx;
    }
    prev_val = head.val;
    idx++;
    head = head.next;
  }
  if (min_dist === Number.MAX_SAFE_INTEGER) min_dist = -1;
  const max_dist = max_point ? max_point - min_point : -1;

  return [min_dist, max_dist];
};
```

### 3 1019. Next Greater Node In Linked List

Medium

You are given the head of a linked list with n nodes.

For each node in the list, find the value of the next greater node. That is, for each node, find the value of the first node that is next to it and has a strictly larger value than it.

Return an integer array answer where answer[i] is the value of the next greater node of the ith node (1-indexed). If the ith node does not have a next greater node, set answer[i] = 0.

Example 1:

Input: head = [2,1,5]
Output: [5,5,0]

Example 2:

Input: head = [2,7,4,3,5]
Output: [7,0,5,5,0]

method 1: 2 loops, very slow O(n^2);

```js
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @return {number[]}
 */
var nextLargerNodes = function (head) {
  let res = [];
  while (head) {
    let temp = head.next;
    let curr_val = head.val;
    while (temp) {
      if (temp.val > curr_val) {
        res.push(temp.val);
        break;
      }
      temp = temp.next;
    }
    if (!temp) res.push(0);
    head = head.next;
  }
  return res;
};
```

### method 2: O(N) with stack

```js
var nextLargerNodes = function (head) {
  let res = [],
    stack = [],
    idx = 0;
  while (head) {
    while (stack.length > 0 && stack.slice(-1)[0][0] < head.val) {
      let i = stack.pop()[1];
      res[i] = head.val;
    }
    res.push(0);
    stack.push([head.val, idx]);
    idx += 1;
    head = head.next;
  }
  return res;
};
```

### 4 1669. Merge In Between Linked Lists

Medium

You are given two linked lists: list1 and list2 of sizes n and m respectively.

Remove list1's nodes from the ath node to the bth node, and put list2 in their place.

The blue edges and nodes in the following figure indicate the result:

Build the result list and return its head.

Example 1:

Input: list1 = [0,1,2,3,4,5], a = 3, b = 4, list2 = [1000000,1000001,1000002]
Output: [0,1,2,1000000,1000001,1000002,5]
Explanation: We remove the nodes 3 and 4 and put the entire list2 in their place. The blue edges and nodes in the above figure indicate the result.

Example 2:

Input: list1 = [0,1,2,3,4,5,6], a = 2, b = 5, list2 = [1000000,1000001,1000002,1000003,1000004]
Output: [0,1,1000000,1000001,1000002,1000003,1000004,6]
Explanation: The blue edges and nodes in the above figure indicate the result.

```js
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} list1
 * @param {number} a
 * @param {number} b
 * @param {ListNode} list2
 * @return {ListNode}
 */
var mergeInBetween = function (list1, a, b, list2) {
  let idx = 1; // find pointer_a before a
  const head = list1;
  let pointer_a = null;
  let pointer_b = null;
  while (list1) {
    if (idx === a) {
      pointer_a = list1;
    }
    if (idx - 1 === b) {
      // find pointer_b before b
      pointer_b = list1;
    }
    list1 = list1.next;
    idx++;
  }
  // intert b to a
  pointer_a.next = list2;
  while (list2.next) {
    list2 = list2.next;
  }
  list2.next = pointer_b.next;
  return head;
};
```

### 5 725. Split Linked List in Parts

Medium

Given the head of a singly linked list and an integer k, split the linked list into k consecutive linked list parts.

The length of each part should be as equal as possible: no two parts should have a size differing by more than one. This may lead to some parts being null.

The parts should be in the order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal to parts occurring later.

Return an array of the k parts.

Example 1:

Input: head = [1,2,3], k = 5
Output: [[1],[2],[3],[],[]]
Explanation:
The first element output[0] has output[0].val = 1, output[0].next = null.
The last element output[4] is null, but its string representation as a ListNode is [].

Example 2:

Input: head = [1,2,3,4,5,6,7,8,9,10], k = 3
Output: [[1,2,3,4],[5,6,7],[8,9,10]]
Explanation:
The input has been split into consecutive parts with size difference at most 1, and earlier parts are a larger size than the later parts.

```js
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @param {number} k
 * @return {ListNode[]}
 */
var splitListToParts = function (head, k) {
  let cur = head;
  let N = 0;

  while (cur != null) {
    cur = cur.next;
    N++;
  }

  const width = Math.floor(N / k),
    rem = N % k;

  let ans = [];
  cur = head; //for traversing whole link list

  for (let i = 0; i < k; i++) {
    let temp = cur; //for pushing the head in the ans array
    // check how many numbers in a section
    for (let j = 0; j < width + (i < rem ? 1 : 0) - 1; ++j) {
      if (cur != null) cur = cur.next;
    }
    // for pointing to null
    if (cur != null) {
      let prev = cur;
      cur = cur.next;
      prev.next = null;
    }
    ans.push(temp);
  }
  return ans;
};
```

### 6 1721. Swapping Nodes in a Linked List

Medium

You are given the head of a linked list, and an integer k.

Return the head of the linked list after swapping the values of the kth node from the beginning and the kth node from the end (the list is 1-indexed).

Example 1:

Input: head = [1,2,3,4,5], k = 2
Output: [1,4,3,2,5]

Example 2:

Input: head = [7,9,6,6,7,8,3,0,9,5], k = 5
Output: [7,9,6,6,8,7,3,0,9,5]
Example 3:

Input: head = [1], k = 1
Output: [1]

Example 4:

Input: head = [1,2], k = 1
Output: [2,1]

```js
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @param {number} k
 * @return {ListNode}
 */
var swapNodes = function (head, k) {
  let length = 0,
    count = 1;
  let a = null,
    b = null;
  let temp = head;
  while (temp) {
    temp = temp.next;
    length++;
  }
  temp = head;
  while (temp) {
    if (count === k) a = temp;
    if (count === length - k + 1) b = temp;
    count++;
    temp = temp.next;
  }
  let curr = a.val;
  a.val = b.val;
  b.val = curr;
  return head;
};
```
