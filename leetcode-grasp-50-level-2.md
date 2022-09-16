#####################################################################################################
The second version of grasp 50 and it is level 2: entry level, there is a few hard questions and
most of them will be easy and medium questions.
#####################################################################################################

### 1 dp nlog(n)

300. Longest Increasing Subsequence

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        def binarySearch(res):
            l, r = 0, len(res) - 1
            while l < r:
                m = (l + r) // 2
                if res[m] < n:
                    l += 1
                else:
                    r = m
            return l
        res = []
        for n in nums:
            if not res or n > res[-1]:
                res.append(n)
            else:
                stackIndex = binarySearch(res)
                res[stackIndex] = n
        return len(res)
```

### 2 dp

### 12

354. Russian Doll Envelopes

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key = lambda x: (x[0], -x[1]))
        nums = []
        for item in envelopes:
            nums.append(item[1])

        # LIS random sort
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

### 3 sets

36. Valid Sudoku

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = defaultdict(set) # 9 sets of rows
        cols = defaultdict(set) # 9 sets of cols
        squares = defaultdict(set) # 9 sets of squares = (r /3, c /3)
        for r in range(9):
            for c in range(9):
                if (board[r][c] in rows[r] or board[r][c] in cols[c] or board[r][c] in squares[(r // 3, c // 3)]):
                    return False
                elif board[r][c] != ".":
                    cols[c].add(board[r][c])
                    rows[r].add(board[r][c])
                    squares[(r // 3, c // 3)].add(board[r][c])
        return True
```

### 4 backtracking

37. Sudoku Solver

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows, cols, block, seen = defaultdict(set), defaultdict(set), defaultdict(set), deque([])
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    rows[i].add(board[i][j])
                    cols[j].add(board[i][j])
                    block[(i // 3,j // 3)].add(board[i][j])
                else:
                    seen.append((i,j))

        def dfs():
            if not seen:
                return True

            r,c = seen[0]
            t = (r//3,c//3)
            for n in {'1','2','3','4','5','6','7','8','9'}:
                if n not in rows[r] and n not in cols[c] and n not in block[t]:
                    board[r][c]=n
                    rows[r].add(n)
                    cols[c].add(n)
                    block[t].add(n)
                    seen.popleft()
                    if dfs():
                        return True
                    else:
                        board[r][c]="."
                        rows[r].discard(n)
                        cols[c].discard(n)
                        block[t].discard(n)
                        seen.appendleft((r,c))
            return False

        dfs()
```

### 5 backtracking

51. N-Queens

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        col, posDiag, negDiag = set(), set(), set()
        res = []
        board = [["."] * n for i in range(n)]

        def backtrack(r):
            if r == n:
                copy = ["".join(row) for row in board]
                res.append(copy)
                return
            for c in range(n):
                if c in col or (r + c) in posDiag or (r - c) in negDiag:
                    continue

                col.add(c)
                posDiag.add(r + c)
                negDiag.add(r - c)
                board[r][c] = "Q"
                backtrack(r + 1)
                col.remove(c)
                posDiag.remove(r + c)
                negDiag.remove(r - c)
                board[r][c] = "."
        backtrack(0)
        return res
```

### 6 backtracking

52. N-Queens II

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        col, posDiag, negDiag, res = set(), set(), set(), []
        board = [["."] * n for i in range(n)]

        def backtrack(r):
            if r == n:
                copy = ["".join(row) for row in board]
                res.append(copy)
            for c in range(n):
                if c not in col and (r + c) not in posDiag and (r - c) not in negDiag:
                    col.add(c)
                    posDiag.add(r + c)
                    negDiag.add(r - c)
                    board[r][c] = "Q"
                    backtrack(r + 1)
                    col.remove(c)
                    posDiag.remove(r + c)
                    negDiag.remove(r - c)
                    board[r][c] = "."
        backtrack(0)
        return len(res)
```

### 7 reverse number

279. Perfect Squares

```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [0] + [n] * n
        for i in range(1, n + 1):
            res = []
            for j in range(1, i + 1):
                if i - j * j < 0:
                    break
                res.append(1 + dp[i - j * j])
            dp[i] = min(res)
        return dp[-1]
```

### 8 reverse number

231. Power of Two

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n <= 0: return False
        res = 0
        while n:
            res += n & 1
            n >>= 1
        return res == 1
# Follow up: Could you solve it without loops/recursion?
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n==1:
            return True
        elif n != 0 and n & (n-1) == 0:
            return True
        else:
            return False
```

### 9 prefix sum

209. Minimum Size Subarray Sum

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l, total = 0, 0
        res = float("inf")
        for r in range(len(nums)):
            total += nums[r]
            while total >= target:
                res = min(res, r - l + 1)
                total -= nums[l]
                l += 1
        return 0 if res == float("inf") else res
```

### 10 prefix sum

665. Non-decreasing Array

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        count = 0
        for i in range(len(nums) - 1):
            if nums[i] <= nums[i + 1]:
                continue
            if i == 0 or nums[i + 1] >= nums[i - 1]:
                nums[i] = nums[i + 1]
            else:
                nums[i + 1] = nums[i]
            count += 1
            if count == 2:
                return False
        return count <= 1
```

### 11 1d dp

792. Number of Matching Subsequences

```python
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        def isSubsequence(s, t): # s is a subsequence of t
            i, j = 0, 0
            while i < len(s) and j < len(t):
                if s[i] == t[j]:
                    i += 1
                    j += 1
                else:
                    j += 1
            return i == len(s)
        cache = {}
        res = 0
        for w in words:
            if w in cache:
                if cache[w]:
                    res += 1
                continue
            if isSubsequence(w, s):
                cache[w] = True
                res += 1
            else:
                cache[w] = False
        return res
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

#####################################################################################################

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

### 1 monotonic stack

739. Daily Temperatures

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res, stack = [0] * len(temperatures), []
        for i in range(len(temperatures)):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                last = stack.pop()
                res[last] = i - last
            stack.append(i)
        return res
```

962. Maximum Width Ramp

```python
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        res, stack, n = 0, [], len(nums)
        for i in range(n):
            if not stack or nums[stack[-1]] > nums[i]:
                stack.append(i)
        for i in range(n - 1, -1, -1):
            while stack and nums[i] >= nums[stack[-1]]:
                last = stack.pop()
                res = max(res, i - last)
        return res
```

84. Largest Rectangle in Histogram

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # non-decreasing stack, res is max area
        res, stack = 0, []
        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                res = max(res, height * (i - index))
                start = index
            stack.append((start, h))
        # handle remaining stack:
        for i, h in stack:
            res = max(res, h * (len(heights) - i))
        return res
```

85. Maximal Rectangle

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        def histHelper(heights):
            res, stack = 0, []
            for i, h in enumerate(heights):
                start = i
                while stack and stack[-1][1] > h:
                    index, height = stack.pop()
                    res = max(res, height * (i - index))
                    start = index
                stack.append((start, h))
            # handle remaining stack:
            for i, h in stack:
                res = max(res, h * (len(heights) - i))
            return res
        # calculate each row
        res = 0
        heights = [0] * len(matrix[0])
        for row in matrix:
            for i in range(len(heights)):
                heights[i] = heights[i] + 1 if row[i] == '1' else 0
            res = max(res, histHelper(heights))
        return res
```

767. Reorganize String

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        count = Counter(s)
        maxHeap = [[-cnt, char] for char, cnt in count.items()]
        heapq.heapify(maxHeap)

        res, prev = '', None
        while maxHeap:
            cnt, char = heapq.heappop(maxHeap)
            res += char
            cnt += 1
            if prev:
                heapq.heappush(maxHeap, prev)
                prev = None
            if cnt != 0:
                prev = [cnt, char]
        return '' if prev else res
```
