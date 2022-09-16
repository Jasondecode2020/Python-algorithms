#####################################################################################################
The first version of grasp 50 and it is level 1: entry level, there is no hard questions and
most of them will be easy questions.
#####################################################################################################

## 200 is similar to 547

### 1 dfs

200. Number of Islands

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grid, r, c):
            grid[r][c] = '0'
            points = [[r - 1, c], [r + 1, c], [r, c + 1], [r, c - 1]]
            for row, col in points:
                if row >= 0 and col >= 0 and row < len(grid) and col < len(grid[0]) and grid[row][col] == '1':
                    dfs(grid, row, col)

        count = 0
        ROWS, COLS = len(grid), len(grid[0])
        for i in range(ROWS):
            for j in range(COLS):
                if grid[i][j] == '1':
                    dfs(grid, i, j)
                    count += 1
        return count
```

### 1 dfs

547. Number of Provinces

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def dfs(start):
            visited.add(start)
            for end in range(len(isConnected)):
                if isConnected[start][end] and end not in visited:
                    dfs(end)

        numberOfProvinces = 0
        visited = set()
        for start in range(len(isConnected)):
            if start not in visited:
                dfs(start)
                numberOfProvinces += 1
        return numberOfProvinces
```

## 2, 3 and 4 are similar

### 2 dfs

94. Binary Tree Inorder Traversal

```python
# recursive
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, arr):
            if not node:
                return node
            dfs(node.left, arr)
            arr.append(node.val)
            dfs(node.right, arr)
            return arr
        return dfs(root, [])
# iterative
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res, stack = [], []
        while True:
            while root:
                stack.append(root)
                root = root.left
            if not stack:
                return res
            node = stack.pop()
            res.append(node.val)
            root = node.right
```

### 3 dfs

144. Binary Tree Preorder Traversal

```python
# recursive
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, arr):
            if not node:
                return node
            arr.append(node.val)
            dfs(node.left, arr)
            dfs(node.right, arr)
            return arr
        return dfs(root, [])
# iterative
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res, stack = [], [root]
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.extend([node.right, node.left])
        return res
```

### 4 dfs

145. Binary Tree Postorder Traversal

```python
# recursive
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, arr):
            if not node:
                return node
            dfs(node.left, arr)
            dfs(node.right, arr)
            arr.append(node.val)
            return arr
        return dfs(root, [])
# iterative
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # modified preorder
        res, stack = [], [root]
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.extend([node.left, node.right])
        return res[::-1]
```

### 5 hash table similar to Contains Duplicate II

1. Two Sum

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for idx, val in enumerate(nums):
            res = target - val
            if res in d:
                return [d[res], idx]
            d[val] = idx
```

### 6 linked list

21. Merge Two Sorted Lists

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
        while list1 and list2:
            if list1.val > list2.val:
                p.next = ListNode(list2.val)
                list2 = list2.next
            else:
                p.next = ListNode(list1.val)
                list1 = list1.next
            p = p.next
        p.next = list1 or list2
        return dummy.next
```

## 7 and 8 are similar

### 7 reverse number

9. Palindrome Number

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False

        def reverseNoneNegativeNumber(x):
            res = 0
            while x:
                res = 10 * res + x % 10
                x //= 10
            return res
        return reverseNoneNegativeNumber(x) == x
```

### 8 reverse number

7. Reverse Integer

```python
class Solution:
    def reverse(self, x: int) -> int:
        def reverseNoneNegativeNumber(x):
            res = 0
            while x:
                res = 10 * res + x % 10
                x //= 10
            return res
        res = reverseNoneNegativeNumber(x) if x > 0 else -reverseNoneNegativeNumber(-x)
        if res <= 2 ** 31 - 1 and res >= -2 ** 31:
            return res
        return 0
```

## 9 and 10 are similar

### 9 prefix sum

1480. Running Sum of 1d Array

```python
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        return nums
```

### 10 prefix sum

724. Find Pivot Index

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        right_sum, left_sum = sum(nums), 0
        for i in range(len(nums)):
            right_sum -= nums[i]
            if right_sum == left_sum:
                return i
            left_sum += nums[i]
        return -1
```

### 11 1d dp

509. Fibonacci Number

```python
class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        first, second = 0, 1
        for i in range(2, n + 1):
            second, first = second + first, second
        return second
```

### 12 1d dp

1137. N-th Tribonacci Number

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        if n < 2:
            return n
        if n == 2:
            return 1
        first, second, third = 0, 1, 1
        for i in range(3, n + 1):
            third, second, first = third + second + first, third, second
        return third
```

### 13 1d dp

70. Climbing Stairs

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 2:
            return n
        first, second = 1, 2
        for i in range(3, n + 1):
            second, first = second + first, second
        return second
```

### 14 1d dp

746. Min Cost Climbing Stairs

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        for i in range(2, len(cost)):
            cost[i] += min(cost[i - 1], cost[i - 2])
        return min(cost[-2:])
```

### 15 1d dp

198. House Robber

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        nums = [0] + nums
        for i in range(2, len(nums)):
            nums[i] = max(nums[i - 1], nums[i - 2] + nums[i])
        return nums[-1]
```

### 16 2 pointers

11. Container With Most Water

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        res = 0
        while l < r:
            res = max(res, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return res
```

### 17 stack

20. Valid Parentheses

```python
class Solution:
    def isValid(self, s: str) -> bool:
        def validPair(l, r):
            d = {"(": ")", "{": "}", "[": "]"}
            if l in d and r == d[l]:
                return True
            return False

        stack = []
        for c in s:
            if stack and validPair(stack[-1], c):
                stack.pop()
            else:
                stack.append(c)
        return not stack
```

### 18 linked list

2. Add Two Numbers

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
        carry = 0
        while l1 or l2:
            val = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
            p.next = ListNode(val % 10)
            p = p.next
            if l1: l1 = l1.next
            if l2: l2 = l2.next
            carry = val // 10
        if carry: p.next = ListNode(1)
        return dummy.next
```

### 19 hash table

205. Isomorphic Strings

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        def checkIsomorphic(a, b, d):
            for i in range(len(a)):
                if a[i] not in d:
                    d[a[i]] = b[i]
                elif d[a[i]] != b[i]:
                    return False
            return True
        return checkIsomorphic(s, t, {}) and checkIsomorphic(t, s, {})
```

### 20 subsequence

392. Is Subsequence

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                j += 1
        return i == len(s)
```

Follow up: Suppose there are lots of incoming s, say s1, s2, ..., sk where k >= 109, and you want to check one by one to see if t has its subsequence. In this scenario, how would you change your code?

### 21 reverse linked list

206. Reverse Linked List

```python
# recursive
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def helper(prev, curr):
            if not curr:
                return prev
            nxt = curr.next
            curr.next = prev
            return helper(curr, nxt)

        return helper(None, head)

# iterative
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            nxt = curr.next
            curr.next = prev
            prev, curr = curr, nxt
        return prev
```

### 22 slow fast pointer

876. Middle of the Linked List

```python
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```

### 23 slow fast pointer

141. Linked List Cycle

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

### 24 slow fast pointer

142. Linked List Cycle II

```python
# the first comment in the below link
# https://leetcode.com/problems/linked-list-cycle-ii/discuss/44822/Java-two-pointer-solution.
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                slow = head
                while True:
                    if slow == fast:
                        return slow
                    slow = slow.next
                    fast = fast.next
        return None
```

### 25 one loop

121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowest, profit = prices[0], 0
        for p in prices:
            lowest = min(lowest, p)
            profit = max(profit, p - lowest)
        return profit
```

### 26 design array

12. Integer to Roman

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        symbol = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        value = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        res = ''
        for i, v in enumerate(value):
            res += num // v * symbol[i]
            num %= v
        return res
```

### 27 design array

13. Roman to Integer

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        d = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        res = 0
        for i in range(len(s)):
            res += d[s[i]]
            if i > 0 and d[s[i]] > d[s[i - 1]]:
                res -= 2 * d[s[i - 1]]
        return res
```

### 28 1d dp meet circle

213. House Robber II

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        nums1, nums2 = [0] + nums[: -1], [0] + nums[1: ] # choose first and not choose first
        for i in range(2, len(nums1)):
            nums1[i] = max(nums1[i - 1], nums1[i - 2] + nums1[i])
            nums2[i] = max(nums2[i - 1], nums2[i - 2] + nums2[i])
        return max(nums1[-1], nums2[-1])
```

### 29 1d dp similar to rob houses

740. Delete and Earn

```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        dp = [0] * (max(nums)+1) # dp index is houses 0, 1, 2, ...
        for n in nums:
            dp[n] += n
        for i in range(2, len(dp)):
            dp[i] = max(dp[i - 1], dp[i] + dp[i - 2])
        return dp[-1]
```

### 30 binary search

704. Binary Search

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                return m
        return -1
```

### 31 binary search

374. Guess Number Higher or Lower

```python
class Solution:
    def guessNumber(self, n: int) -> int:
        l, r = 1, n
        while l <= r:
            m = (l + r) // 2
            if guess(m) > 0:
                l = m + 1
            elif guess(m) < 0:
                r = m - 1
            else:
                return m
```

### 32 binary search

278. First Bad Version

```python
class Solution:
    def firstBadVersion(self, n: int) -> int:
        l, r = 1, n
        while l <= r:
            m = (l + r) // 2
            if not isBadVersion(m):
                l = m + 1
            else:
                r = m - 1
        return r + 1
```

### 33 Tree

589. N-ary Tree Preorder Traversal

```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        res = [root.val]
        if root.children:
            for child in root.children:
                res += self.preorder(child)
        return res
```

### 34 Tree

590. N-ary Tree Preorder Traversal

```python
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        if not root:
            return root
        res = []
        if root.children:
            for child in root.children:
                res += self.postorder(child)
        res += [root.val]
        return res
```

### 35 Tree

102. Binary Tree Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        queue, res = deque([root]), []

        while queue:
            level = []
            for i in range(len(queue)): # level count
                node = queue.popleft()
                level.append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(level)
        return res
```

### 36 panlindrom

409. Longest Palindrome

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        d = Counter(s)
        res = odd = 0
        for c in d:
            if d[c] % 2 == 0:
                res += d[c]
            else:
                res += d[c] - 1
                odd += 1
        return res if odd == 0 else res + 1
```

### 37 prefix

14. Longest Common Prefix

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = strs[0]
        for item in strs:
            while not item.startswith(res):
                res = res[:-1]
        return res
```

### 38 Remove Duplicates

26. Remove Duplicates from Sorted Array

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        k = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[k] = nums[i]
                k += 1
        return k
```

### 39 Remove Duplicates

27. Remove Element

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            if nums[l] == val:
                nums[l] = nums[r]
                r -= 1
            else:
                l += 1
        return r + 1
```

### 40 Remove Duplicates

83. Remove Duplicates from Sorted List

```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        p = head
        while p.next:
            if p.val == p.next.val:
                p.next = p.next.next
            else:
                p = p.next
        return head
```

### 41 Duplicate

217. Contains Duplicate

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        s = set()
        for n in nums:
            if n in s:
                return True
            s.add(n)
        return False
```

### 42 similar to 2 sum

219. Contains Duplicate II

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        dic = {}
        for i, v in enumerate(nums):
            if v in dic and i - dic[v] <= k:
                return True
            dic[v] = i
        return False
```

### 43 Remvoe elements

203. Remove Linked List Elements

```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        p = dummy = ListNode()
        p.next = head
        while p.next:
            if p.next.val == val:
                p.next = p.next.next
            else:
                p = p.next
        return dummy.next
```

### 44 Remove Nth Node

19. Remove Nth Node From End of List

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        slow = fast = dummy = ListNode()
        dummy.next = head
        for i in range(n + 1):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
```

### 45 Array

58. Length of Last Word

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        s = s.strip()
        return len(s.split(" ")[-1])
```

### 46 binary search

35. Search Insert Position

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                return m
        return l
```

### 47 strStr

28. Implement strStr()

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack)):
            if haystack[i:].startswith(needle):
                return i
        return -1
```

### 48 Plus One

66. Plus One

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if digits[-1] != 9:
            digits[-1] = digits[-1] + 1
        else:
            res = int(''.join([str(item) for item in digits])) + 1
            digits = [int(i) for i in list(str(res))]
        return digits
```

### 49 Anagram

242. Valid Anagram

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)
```

### 50 Palindrome

125. Valid Palindrome

```python
cclass Solution:
    def isPalindrome(self, s: str) -> bool:
        def helper(s):
            for i in range(len(s) // 2):
                if s[i] != s[len(s) - i - 1]:
                    return False
            return True
        res = ''
        for c in s:
            if c.isalnum():
                res += c.lower()
        return helper(res)
```
