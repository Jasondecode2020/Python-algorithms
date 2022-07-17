### 1

21. Merge Two Sorted Lists

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        p = dummy
        while list1 and list2:
            if list1.val < list2.val:
                p.next = ListNode(list1.val)
                list1 = list1.next
            else:
                p.next = ListNode(list2.val)
                list2 = list2.next
            p = p.next
        p.next = list1 or list2
        return dummy.next
```

### 2

1480. Running Sum of 1d Array

```python
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        return nums
```

### 3

724. Find Pivot Index

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        left, right = nums.copy(), nums.copy()
        right.reverse()
        for i in range(1, len(nums)):
            left[i] += left[i - 1]
            right[i] += right[i - 1]
        right.reverse()
        for i in range(len(nums)):
            if left[i] == right[i]:
                return i
        return -1
```

### 4

205. Isomorphic Strings

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        def checkIsomorphic(a, b, d):
            for i in range(len(a)):
                if a[i] not in d:
                    d[a[i]] = b[i]
                else:
                    if d[a[i]] != b[i]:
                        return False
            return True
        return checkIsomorphic(s, t, {}) and checkIsomorphic(t, s, {})
```

### 5

392. Is Subsequence

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0
        while (i < len(s) and j < len(t)):
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                j += 1
        return i == len(s)
```

### 6

206. Reverse Linked List

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev
```

### 7

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

### 8

142. Linked List Cycle II

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        s = set()
        while head:
            if head in s:
                return head
            s.add(head)
            head = head.next
        return None
```

```python
# https://leetcode.com/problems/linked-list-cycle-ii/discuss/44822/Java-two-pointer-solution.
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                slow = head
                while slow:
                    if slow == fast:
                        return slow
                    slow = slow.next
                    fast = fast.next
        return None
```

### 9

121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowest = prices[0]
        profit = 0
        for i in range(1, len(prices)):
            lowest = min(lowest, prices[i])
            profit = max(profit, prices[i] - lowest)
        return profit
```

### 10

409. Longest Palindrome

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        d = Counter(s)
        res = 0
        odd = 0
        for c in d:
            if d[c] % 2 == 0:
                res += d[c]
            else:
                res += d[c] - 1
                odd += 1
        return res if odd == 0 else res + 1
```

### 11

300. Longest Increasing Subsequence

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        res = []
        for n in nums:
            if not res or n > res[-1]:
                res.append(n)
            else:
                l, r = 0, len(res) - 1
                while l < r:
                    m = (l + r) // 2
                    if res[m] < n:
                        l += 1
                    else:
                        r = m
                res[l] = n
        return len(res)
```

### 12

354. Russian Doll Envelopes

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key = lambda x: (x[0], -x[1]))
        nums = []
        for item in envelopes:
            nums.append(item[1])

        # LIS
        res = []
        for n in nums:
            if not res or n > res[-1]:
                res.append(n)
            else:
                l, r = 0, len(res) - 1
                while l < r:
                    m = (l + r) // 2
                    if res[m] < n:
                        l += 1
                    else:
                        r = m
                res[l] = n
        return len(res)
```

### 13

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

### 14

102. Binary Tree Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue, result = deque([root]), []

        while queue:
            level = []
            for i in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(level)
        return result
```

### 15

509. Fibonacci Number

```python
class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        first, second = 0, 1
        for i in range(2, n + 1):
            first, second = second, first + second
        return second
```

### 16

70. Climbing Stairs

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = [1 for i in range(n + 1)]
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]
```

### 17

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

### 18

278. First Bad Version

```python
class Solution:
    def firstBadVersion(self, n: int) -> int:
        l, r = 1, n
        while l < r:
            m = (l + r) // 2
            if not isBadVersion(m):
                l = m + 1
            else:
                r = m
        return r
```

### 19

1094. Car Pooling
      method 1

```python
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        trips.sort(key = lambda x : x[1])
        minHeap = [] # python used min heap and used index 0 for heap calculation
        curr = 0
        for t in trips:
            num, start, end = t
            while minHeap and minHeap[0][0] <= start:
                curr -= minHeap[0][1]
                heapq.heappop(minHeap)
            curr += num
            if curr > capacity:
                return False
            heapq.heappush(minHeap, [end, num])
        return True
```

method 2

```python
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        change = [0 for i in range(1001)]
        for t in trips:
            num, start, end = t
            change[start] += num
            change[end] -= num
        curr = 0
        for i in range(1001):
            curr += change[i]
            if curr > capacity:
                return False
        return True
```

### 20

1638. Count Substrings That Differ by One Character

```python
class Solution:
    def countSubstrings(self, s: str, t: str) -> int:
        # count all numbers
        res = 0
        # brute force check all conditions
        for i in range(len(s)):
            for j in range(len(t)):
                x, y = i, j
                # check when to count res and when to stop
                count = 0
                while x < len(s) and y < len(t):
                    if s[x] != t[y]:
                        count += 1
                    if count == 1:
                        res += 1
                    if count == 2:
                        break
                    x += 1
                    y += 1
        return res
```

### 21

1. Two Sum

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, val in enumerate(nums):
            res = target - val
            if res in d:
                return [d[res], i]
            d[val] = i
```

### 22

299. Bulls and Cows

```python
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        bulls = 0
        bucket = [0 for i in range(10)]
        for s, g in zip(secret, guess):
            if s == g:
                bulls += 1
            else:
                bucket[int(s)] += 1
                bucket[int(g)] -= 1
        return f'{bulls}A{len(secret) - bulls - sum(x for x in bucket if x > 0)}B'
```

### 23

475. Heaters

```python
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        def closest(heaters, house):
            l, r = 0, len(heaters) - 1
            min_dist = float('inf')
            while l <= r:
                m = (l + r) // 2
                min_dist = min(min_dist, abs(heaters[m] - house))
                if heaters[m] < house:
                    l = m + 1
                else:
                    r = m - 1
            return min_dist

        radius = 0
        heaters.sort()
        for house in houses:
            radius = max(radius, closest(heaters, house))
        return radius
```

### 24

403. Frog Jump

```python
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        n = len(stones)
        stoneSet = set(stones)
        visited = set()
        def goFurther(value, units):
            if (value + units not in stoneSet) or ((value,units) in visited):
                return False
            if value + units == stones[n-1]:
                return True
            visited.add((value,units))
            return goFurther(value + units,units) or goFurther(value + units,units - 1) or goFurther(value + units,units + 1)
        return goFurther(stones[0], 1)
```

### 25

658. Find K Closest Elements

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        n = len(arr)
        start = 0
        end = n - k
        while start < end:
            mid = (start + end) // 2
            if x - arr[mid] > arr[mid + k] - x:  # move right
                start = mid + 1
            else:
                end = mid
        return arr[start: start + k]
```

### 26 382. Linked List Random Node

https://leetcode.com/problems/linked-list-random-node/discuss/1672358/C%2B%2BPythonJava-Reservoir-sampling-oror-Prove-step-by-step-oror-Image

```python
class Solution:

    def __init__(self, head: Optional[ListNode]):
        self.nodes = []
        while head:
            self.nodes.append(head.val)
            head = head.next
    def getRandom(self) -> int:
        return random.choice(self.nodes)
```

### 27 817. Linked List Components

```python
class Solution:
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        s = set(nums)
        connected = False
        count = 0
        while head:
            if head.val in s and not connected:
                count += 1
                connected = True
            elif not head.val in s and connected:
                connected = False
            head = head.next
        return count
```

### 29 925. Long Pressed Name

```python
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        def getFrequencyArray(name):
            arr = []
            i = 0
            count = 1
            while i + 1 < len(name):
                if name[i] == name[i + 1]:
                    count += 1
                else:
                    arr.append([name[i], count])
                    count = 1
                i += 1
            if arr and name[-1] == arr[-1][0]:
                arr[-1][1] += 1
            else:
                arr.append([name[-1], 1])
            return arr

        nameArr = getFrequencyArray(name)
        typedArr = getFrequencyArray(typed)
        if (len(nameArr) != len(typedArr)):
            return False
        for i in range(len(nameArr)):
            if nameArr[i][0] != typedArr[i][0] or nameArr[i][1] > typedArr[i][1]:
                return False
        return True
```

### 30 859. Buddy Strings

```python
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        # if lengths are different, then must be false
        if len(s) != len(goal):
            return False
        # If s and goal are same, then A must have duplicate character
        if s == goal:
            seen = set()
            for a in s:
                if a in seen:
                    return True
                seen.add(a)
            return False

        pair = []
        # when s and goal are not same
        for a, b in zip(s, goal):
            if a != b:
                pair.append((a, b))
            if len(pair) > 2:
                return False

        return len(pair) == 2 and pair[0] == pair[1][::-1]
```

dfs

1 200. Number of Islands

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

2 94. Binary Tree Inorder Traversal

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

3 144. Binary Tree Preorder Traversal

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

4 145. Binary Tree Postorder Traversal

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

### 11

300. Longest Increasing Subsequence

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        res = []
        for n in nums:
            if not res or n > res[-1]:
                res.append(n)
            else:
                l, r = 0, len(res) - 1
                while l < r:
                    m = (l + r) // 2
                    if res[m] < n:
                        l += 1
                    else:
                        r = m
                res[l] = n
        return len(res)
```

### 12

354. Russian Doll Envelopes

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key = lambda x: (x[0], -x[1]))
        nums = []
        for item in envelopes:
            nums.append(item[1])

        # LIS
        res = []
        for n in nums:
            if not res or n > res[-1]:
                res.append(n)
            else:
                l, r = 0, len(res) - 1
                while l < r:
                    m = (l + r) // 2
                    if res[m] < n:
                        l += 1
                    else:
                        r = m
                res[l] = n
        return len(res)
```

### 19

1094. Car Pooling
      method 1

```python
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        trips.sort(key = lambda x : x[1])
        minHeap = [] # python used min heap and used index 0 for heap calculation
        curr = 0
        for t in trips:
            num, start, end = t
            while minHeap and minHeap[0][0] <= start:
                curr -= minHeap[0][1]
                heapq.heappop(minHeap)
            curr += num
            if curr > capacity:
                return False
            heapq.heappush(minHeap, [end, num])
        return True
```

method 2

```python
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        change = [0 for i in range(1001)]
        for t in trips:
            num, start, end = t
            change[start] += num
            change[end] -= num
        curr = 0
        for i in range(1001):
            curr += change[i]
            if curr > capacity:
                return False
        return True
```

### 20

1638. Count Substrings That Differ by One Character

```python
class Solution:
    def countSubstrings(self, s: str, t: str) -> int:
        # count all numbers
        res = 0
        # brute force check all conditions
        for i in range(len(s)):
            for j in range(len(t)):
                x, y = i, j
                # check when to count res and when to stop
                count = 0
                while x < len(s) and y < len(t):
                    if s[x] != t[y]:
                        count += 1
                    if count == 1:
                        res += 1
                    if count == 2:
                        break
                    x += 1
                    y += 1
        return res
```

### 21

299. Bulls and Cows

```python
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        bulls = 0
        bucket = [0 for i in range(10)]
        for s, g in zip(secret, guess):
            if s == g:
                bulls += 1
            else:
                bucket[int(s)] += 1
                bucket[int(g)] -= 1
        return f'{bulls}A{len(secret) - bulls - sum(x for x in bucket if x > 0)}B'
```

### 23

475. Heaters

```python
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        def closest(heaters, house):
            l, r = 0, len(heaters) - 1
            min_dist = float('inf')
            while l <= r:
                m = (l + r) // 2
                min_dist = min(min_dist, abs(heaters[m] - house))
                if heaters[m] < house:
                    l = m + 1
                else:
                    r = m - 1
            return min_dist

        radius = 0
        heaters.sort()
        for house in houses:
            radius = max(radius, closest(heaters, house))
        return radius
```

### 24

403. Frog Jump

```python
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        n = len(stones)
        stoneSet = set(stones)
        visited = set()
        def goFurther(value, units):
            if (value + units not in stoneSet) or ((value,units) in visited):
                return False
            if value + units == stones[n-1]:
                return True
            visited.add((value,units))
            return goFurther(value + units,units) or goFurther(value + units,units - 1) or goFurther(value + units,units + 1)
        return goFurther(stones[0], 1)
```

### 25

658. Find K Closest Elements

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        n = len(arr)
        start = 0
        end = n - k
        while start < end:
            mid = (start + end) // 2
            if x - arr[mid] > arr[mid + k] - x:  # move right
                start = mid + 1
            else:
                end = mid
        return arr[start: start + k]
```

### 26 382. Linked List Random Node

https://leetcode.com/problems/linked-list-random-node/discuss/1672358/C%2B%2BPythonJava-Reservoir-sampling-oror-Prove-step-by-step-oror-Image

```python
class Solution:

    def __init__(self, head: Optional[ListNode]):
        self.nodes = []
        while head:
            self.nodes.append(head.val)
            head = head.next
    def getRandom(self) -> int:
        return random.choice(self.nodes)
```

### 27 817. Linked List Components

```python
class Solution:
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        s = set(nums)
        connected = False
        count = 0
        while head:
            if head.val in s and not connected:
                count += 1
                connected = True
            elif not head.val in s and connected:
                connected = False
            head = head.next
        return count
```

### 29 925. Long Pressed Name

```python
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        def getFrequencyArray(name):
            arr = []
            i = 0
            count = 1
            while i + 1 < len(name):
                if name[i] == name[i + 1]:
                    count += 1
                else:
                    arr.append([name[i], count])
                    count = 1
                i += 1
            if arr and name[-1] == arr[-1][0]:
                arr[-1][1] += 1
            else:
                arr.append([name[-1], 1])
            return arr

        nameArr = getFrequencyArray(name)
        typedArr = getFrequencyArray(typed)
        if (len(nameArr) != len(typedArr)):
            return False
        for i in range(len(nameArr)):
            if nameArr[i][0] != typedArr[i][0] or nameArr[i][1] > typedArr[i][1]:
                return False
        return True
```

### 30 859. Buddy Strings

```python
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        # if lengths are different, then must be false
        if len(s) != len(goal):
            return False
        # If s and goal are same, then A must have duplicate character
        if s == goal:
            seen = set()
            for a in s:
                if a in seen:
                    return True
                seen.add(a)
            return False

        pair = []
        # when s and goal are not same
        for a, b in zip(s, goal):
            if a != b:
                pair.append((a, b))
            if len(pair) > 2:
                return False

        return len(pair) == 2 and pair[0] == pair[1][::-1]
```

dfs
