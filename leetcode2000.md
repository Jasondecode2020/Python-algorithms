## Guide

1. c: character
2. d: dictionary
3. dummy: dummy node
4. i: index
5. idx: index
6. l: left
7. p: pointer
8. r: right
9. res: final return
10. val: value
11. A: small array
12. B: biger array
13. total: A + B
14. half: half of total
15. Aleft: partition left of A
16. Aright: partition right of A
17. Bleft: partition left of B
18. Bright: partition right of B
19. ans: temp value of res
20. flip: -1 or 1
21. row: row number
22. slow: slow pointer
23. fast: fast pointer
24. pq: priority queue
25. rows: all rows of a grid
26. cols: all rows of a grid
27. squares: an area if grid

## Data structure and algorithm uesd

1. hash table
2. linked list
3. binary search
4. sliding window
5. dp
6. dfs
7. bucket

## Function used

1. lengthOfPalindrome(l, r): length of palindrome start at position l, r
2. dfs: to solve dfs problem
3. bfs: to solve bfs problem
4. reverseNoneNegativeNumber(n): reverse a positive number n or 0

### 1. Two Sum

#### hash table

#### line: 6

#### state: pass

The question is asked to return 2 indexes from the array, the best way is
to use a hash table, this run a time of O(n) and need space O(n) for building
the hash map, and python has the build in date structure dict to solve this
problem. If we can put one value in the hash table, for the other value, it
becomes a search problem in the table.

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

### 2. Add Two Numbers

#### linked list

#### line: 11

#### state: pass

The question is asked to add 2 numbers using linked list, to return the result represented
by a linked list, the dummy node is used to avoid edge cases, often used in linked list, and
carry is used for plus calculation. Just loop through the list and return dummy.next.
carry and dummy are very important in this kind of problem.

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

### 3. Longest Substring Without Repeating Characters

#### sliding window + hash table

#### line: 7

#### state: pass

The question asked for longest sub string, sliding window can solve it in O(n) time, the l index
need to be taken care when the cases like "abbca", l index is not always d[c] + 1. If after the first "a", there has a "bb", it means the l has updated, this need to be careful.

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        d, l, res = {}, 0, 0
        for r, c in enumerate(s):
            if c in d:
                l = max(l, d[c] + 1)
            res = max(res, r - l + 1)
            d[c] = r
        return res
```

### 4. Median of Two Sorted Arrays

#### binary search

#### line: 19

#### state: not pass

The question asked for median of 2 sorted array, and need time O(log(m + n)).
Need to use A array as the smaller array for binary search, and also check edge cases of A[i + 1]
and B[j + 1], Aleft is -inf, Aright is inf, Bleft is -inf and Bright is inf.

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        A, B, total = nums1, nums2, len(nums1) + len(nums2)
        half = total // 2
        if len(A) > len(B): A, B = B, A

        l, r = 0, len(A) - 1
        while True:
            i = (l + r) // 2
            j = half - i - 2
            leftA = A[i] if i >= 0 else float('-inf')
            rightA = A[i + 1] if i + 1 < len(A) else float('inf')
            leftB =B[j] if j >= 0 else float('-inf')
            rightB = B[j + 1] if j + 1 < len(B) else float('inf')
            if leftA <= rightB and leftB <= rightA:
                if total % 2:
                    return min(rightA, rightB)
                return (max(leftA, leftB) + min(rightA, rightB)) / 2
            elif leftA > rightB:
                r = i - 1
            else:
                l = i + 1
```

link: https://www.youtube.com/watch?v=q6IEA26hvXc

### 5. Longest Palindromic Substring

#### two pointers

#### line: 7 + lengthOfPalindrome(l, r): line 5

#### state: pass

The question asked for longest sub panlindrome, the easiest way is brute force
here use 2 pointers brute force.

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def lengthOfPalindrome(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l + 1: r]

        res = ''
        for i in range(len(s)):
            ans = lengthOfPalindrome(i, i)
            if len(ans) > len(res): res = ans
            ans = lengthOfPalindrome(i, i + 1)
            if len(ans) > len(res): res = ans
        return res
```

### 6. Zigzag Conversion

#### bucket sort

#### line: 10

The question asked to bucket sort, put all letters in rows of buckets, if there is another
question used this method, I'll right a function of filling in the bucket.

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s): return s
        bucket, flip, row = [[] for i in range(numRows)], -1, 0
        for c in s:
            bucket[row].append(c)
            if row == 0 or row == numRows - 1:
                flip *= -1
            row += flip
        for i in range(len(bucket)):
            bucket[i] = "".join(bucket[i])
        return "".join(bucket)
```

### 7. Reverse Integer

#### reverse positive number

#### line: 3 + reverseNoneNegativeNumber(n)

The question asked to use reverseNoneNegativeNumber(n), similar to question 9

```python
class Solution:
    def reverse(self, x: int) -> int:
        def reverseNoneNegativeNumber(n):
            res = 0
            while n:
                res = res * 10 + n % 10
                n //= 10
            return res
        res = reverseNoneNegativeNumber(x) if x > 0 else -reverseNoneNegativeNumber(-x)
        if res >= -2 ** 31 and res <= 2 ** 31 - 1: return res
        return 0
```

8. String to Integer (atoi)

#### atoi string to integer

#### line: 12

The question asked for longest sub panlindrome, the easiest way is brute force
here use 2 pointers brute force. The problem is is didn't mention what it is
when res is '', '-' or '+'.

```python
class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()
        sign, res = ['+', '-'], ""
        for i, c in enumerate(s):
            if i == 0 and c in sign:
                res += c
            elif c.isnumeric():
                res += c
            else:
                break
        if not res or res in sign:
            return 0
        return max(min(int(res), 2**31 -1), -2**31)
```

9. Palindrome Number

#### reverse positive number

#### line: 1 + reverseNoneNegativeNumber(n)

The question asked to use reverseNoneNegativeNumber(n), similar to question 7

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        def reverseNoneNegativeNumber(n):
            res = 0
            while n:
                res = res * 10 + n % 10
                n //= 10
            return res
        return False if x < 0 else reverseNoneNegativeNumber(x) == x
```

### 10. Regular Expression Matching

#### Dp = dfs

#### line: 12

The question asked to use match of regular expression, we can use cache or brute force dfs

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        def dfs(i, j):
            if i >= len(s) and j >= len(p):
                return True
            if j >= len(p):
                return False
            match = i < len(s) and (s[i] == p[j] or p[j] == ".")
            if (j + 1) < len(p) and p[j + 1] == "*":
                return (dfs(i, j + 2) or (match and dfs(i + 1, j)))
            if match:
                return dfs(i + 1, j + 1)
            return False
        return dfs(0, 0)
```

#### df + cache = dfs + cache

#### line: 17

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        cache = {}
        def dfs(i, j):
            if (i, j) in cache:
                return cache[(i, j)]
            if i >= len(s) and j >= len(p):
                return True
            if j >= len(p):
                return False
            match = i < len(s) and (s[i] == p[j] or p[j] == ".")
            if (j + 1) < len(p) and p[j + 1] == "*":
                cache[(i, j)] = (dfs(i, j + 2) or (match and dfs(i + 1, j)))
                return cache[(i, j)]
            if match:
                cache[(i, j)] = dfs(i + 1, j + 1)
                return cache[(i, j)]
            return False
        return dfs(0, 0)
```

### need to check 4, 8, 10 next time

### 11. Container With Most Water

#### two pointers

#### line: 8

The question asked to get maxArea, 2 pointers is easier to solve in a loop, this is a
kind of greedy way to solve problems.

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r, res = 0, len(height) - 1, 0
        while l < r:
            res = max(res, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return res
```

### 12. Integer to Roman

#### two pointers

#### line: 7

The question asked to get maxArea, 2 pointers is easier to solve in a loop, this is a
kind of greedy way to solve problems.

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

### 13. Roman to Integer

#### two pointers

#### line: 6 + d

The question asked to get maxArea, 2 pointers is easier to solve in a loop, this is a
kind of greedy way to solve problems.

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
        for i, c in enumerate(s):
            res += d[c]
            if i > 0 and d[c] > d[s[i - 1]]:
                res -= 2 * d[s[i - 1]]
        return res
```

### 14. Longest Common Prefix

#### two pointers

#### line: 8

The question asked to get maxArea, 2 pointers is easier to solve in a loop, this is a
kind of greedy way to solve problems.

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = strs[0]
        for item in strs:
            while not item.startswith(res):
                res = res[:-1]
        return res
```

### 15. 3Sum

#### two pointers

#### line: 16

The question asked to get 3 numbers sum to 0, 2 pointers is easier to solve in O(n^2), this is a
kind of greedy way to solve problems.

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = set()
        nums.sort()
        for i in range(len(nums)):
            l = i + 1
            r = len(nums) - 1
            while l < r:
                three = nums[i] + nums[l] + nums[r]
                if three > 0:
                    r -= 1
                elif three < 0:
                    l += 1
                else:
                    res.add((nums[i], nums[l], nums[r]))
                    r -= 1
                    l += 1
        return res
```

### 16. 3Sum Closest

#### two pointers

#### line: 17

The question asked to get 3 numbers sum to 0, 2 pointers is easier to solve in O(n^2), this is a
kind of greedy way to solve problems.

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        res = float('inf')
        nums.sort()
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                three = nums[i] + nums[l] + nums[r]
                if three == target:
                    return target
                if abs(target - three) < abs(target - res):
                    res = three
                if three > target:
                    r -= 1
                else:
                    l += 1
        return res
```

### 16. 3Sum Closest

#### two pointers

#### line: 9

The question asked to get combinations of all letters of 2 - 9, it's 4 ^ 8 time complexity.

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits: return []
        d = {
                "2": ["a", "b", "c"],
                "3": ["d", "e", "f"],
                "4": ["g", "h", "i"],
                "5": ["j", "k", "l"],
                "6": ["m", "n", "o"],
                "7": ["p", "q", "r", "s"],
                "8": ["t", "u", "v"],
                "9": ["w", "x", "y", "z"],
             }
        res = [''] # [''], ["a", "b", "c"]
        for n in digits: # n = 2, 3
            letters = d[n] # ["a", "b", "c"], ["d", "e", "f"]
            ans = [] # [], ["a", "b", "c"]
            for r in res: # r = '', ["a", "b", "c"]
                for l in letters: # l = 'a', l = 'd'
                    ans.append(r+l) # ["a", "b", "c"] ["ad","ae","af","bd","be","bf","cd","ce","cf"]
            res = ans # ["a", "b", "c"], ["ad","ae","af","bd","be","bf","cd","ce","cf"]
        return res
```

### 18. 4Sum

#### two pointers

#### line: 17

The question asked to get 3 numbers sum to 0, 2 pointers is easier to solve in O(n^2), this is a
kind of greedy way to solve problems.

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        s = set()
        nums.sort()
        for i in range(len(nums) - 1):
            if i > 0 and nums[i] == nums[i-1]: continue
            for j in range(i + 1, len(nums)):
                lo, hi = j + 1, len(nums) - 1
                while lo < hi:
                    four = nums[i] + nums[j] + nums[lo] + nums[hi]
                    if four == target:
                        s.add((nums[i], nums[j], nums[lo], nums[hi]))
                        lo += 1
                        hi -= 1
                    elif four < target:
                        lo += 1
                    else:
                        hi -= 1
        return s
```

### 19. Remove Nth Node From End of List

#### slow fast pointers

#### line: 9

The question asked to remove from the end of nth, we can use math directly. Here
we can slow and fast pointers.

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

### 20. Valid Parentheses

#### stack

#### line: 7 + validPair(left, right)

The question asked to get 3 numbers sum to 0, 2 pointers is easier to solve in O(n^2), this is a
kind of greedy way to solve problems.

```python
class Solution:
    def isValid(self, s: str) -> bool:
        def validPair(left, right):
            d = {"(": ")", "{": "}", "[": "]"}
            if left in d and d[left] == right:
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

### 921. Minimum Add to Make Parentheses Valid

nearly the same as 20

```python
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        def validPair(left, right):
            d = {"(": ")", "{": "}", "[": "]"}
            if left in d and d[left] == right:
                return True
            return False

        stack = []
        for c in s:
            if stack and validPair(stack[-1], c):
                stack.pop()
            else:
                stack.append(c)
        return len(stack)
```

### 21. Merge Two Sorted Lists

#### stack

#### line: 11

The question asked to merge 2 sorted list, the lists are sorted, just merge them, similar to question 2

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
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

### 21. Merge Two Sorted Lists

#### stack

#### line: 11

The question asked to merge 2 sorted list, the lists are sorted, just merge them, similar to question 2

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
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

### 22. Generate Parentheses

#### dfs + stack

#### line: 14

The question asked to merge 2 sorted list, the lists are sorted, just merge them, similar to question 2

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        stack, res = [], [] # hold valid parentheses
        def dfs(openN, closedN):
            if openN == closedN == n:
                res.append(''.join(stack)) # find first result
            if openN < n:
                stack.append('(')
                dfs(openN + 1, closedN)
                stack.pop() # stack is global
            if closedN < openN:
                stack.append(')')
                dfs(openN, closedN + 1)
                stack.pop() # stack is global
        dfs(0, 0)
        return res
```

### 23. Merge k Sorted Lists

#### pq

#### line: 12

The question asked to merge 2 sorted list, the lists are sorted, just merge them, similar to question 2

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        pq, res = [], []
        p = dummy = ListNode()
        for lst in lists:
            temp = lst
            while temp:
                heapq.heappush(pq, temp.val)
                temp = temp.next
        while pq:
            val = heapq.heappop(pq)
            p.next = ListNode(val)
            p = p.next
        return dummy.next
```

### 24. Swap Nodes in Pairs

#### linked list

#### line: 6

The question asked to swap 2 nodes, here used recursive function.

recursive: line 6

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        first, second = head, head.next
        first.next = self.swapPairs(second.next)
        second.next = first
        return second
```

iterative: line: 15

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = dummy = ListNode(0)
        dummy.next, curr, total = head, head, 0
        while head:
            total += 1
            head = head.next

        def reverseOneNode():
            nxt = curr.next
            curr.next = nxt.next
            nxt.next = prev.next
            prev.next = nxt
        for i in range(total // 2):
            reverseOneNode()
            prev = curr
            curr = prev.next
        return dummy.next
```

### 25. Reverse Nodes in k-Group

#### linked list

#### line: 6

The question asked to swap k nodes as a group, here used iterative function.

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        prev = dummy = ListNode(0)
        dummy.next, curr, total = head, head, 0
        while head:
            total += 1
            head = head.next

        def reverseOneNode():
            nxt = curr.next
            curr.next = nxt.next
            nxt.next = prev.next
            prev.next = nxt
        for i in range(total // k):
            for j in range(k - 1):
                reverseOneNode()
            prev = curr
            curr = prev.next
        return dummy.next
```

26. Remove Duplicates from Sorted Array

#### array index

#### line: 6

The question asked to remove duplicates in sorted array

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

### 27. Remove Element

#### two pointers

#### line: 8

The question asked to remove val in an array

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
        return l
```

### 28. Implement strStr()

#### string

#### line: 4

The question asked to use startswith

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack)):
            if haystack[i:].startswith(needle):
                return i
        return -1
```

### 31. Next Permutation

#### string

#### line: 4

The question asked to use startswith

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        def reverseLastPart(l, r):
            while l < r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1
        i = j = len(nums) - 1
        while i > 0 and nums[i - 1] >= nums[i]:
            i -= 1
        if i == 0:   # nums are in descending order
            nums.reverse()
            return
        k = i - 1    # find the last "ascending" position
        while nums[j] <= nums[k]:
            j -= 1
        nums[k], nums[j] = nums[j], nums[k]
        reverseLastPart(k + 1, len(nums) - 1)  # reverse the second part
```

### 32. Longest Valid Parentheses

#### stack

#### line: 11

The question asked to use startswith

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack, res = [-1], 0 # edge cases of s = "()"
        for i in range(len(s)):
            if s[i] == "(": # prepare for finding the max
                stack.append(i)
            else:
                stack.pop()
                if len(stack) == 0:
                    stack.append(i)
                else:
                    res = max(res, i - stack[-1])
        return res
```

### 36. Valid Sudoku

#### set

#### line: 10

The question asked to check if there is a solution for sudoku, need to use 3 sets store the values
for each loop of every point, then check if there is a false before loop through all board. If no false,
return true.

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows, cols, squares = defaultdict(set), defaultdict(set), defaultdict(set)
        for r in range(9):
            for c in range(9):
                if (board[r][c] in (rows[r] | cols[c] | squares[(r // 3, c // 3)])):
                    return False
                elif board[r][c] != ".":
                    cols[c].add(board[r][c])
                    rows[r].add(board[r][c])
                    squares[(r // 3, c // 3)].add(board[r][c])
        return True
```

### 37. Sudoku Solver

#### dfs

#### line: 11

The question asked to use dfs + set

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows, cols, squares, visit = defaultdict(set), defaultdict(set), defaultdict(set), deque([])
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    rows[i].add(board[i][j])
                    cols[j].add(board[i][j])
                    squares[(i // 3, j // 3)].add(board[i][j])
                else:
                    visit.append((i,j))

        def dfs():
            if not visit:
                return True

            r, c = visit[0]
            square, numbers = (r // 3, c // 3), {'1','2','3','4','5','6','7','8','9'}
            for n in numbers: # try 9 ways
                if n not in (rows[r] | cols[c] | squares[square]):
                    board[r][c] = n
                    rows[r].add(n)
                    cols[c].add(n)
                    squares[square].add(n)
                    visit.popleft()
                    if dfs(): # find 1 way
                        return True
                    else: # backtrack
                        board[r][c] = "."
                        rows[r].discard(n)
                        cols[c].discard(n)
                        squares[square].discard(n)
                        visit.appendleft((r,c))
            return False # not find
        dfs()
```

### 46. Permutations

#### set

#### line: 10

The question asked to check if there is a solution for sudoku, need to use 3 sets store the values
for each loop of every point, then check if there is a false before loop through all board. If no false,
return true.

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        if len(nums) == 1:
            return [nums[:]]
        for i in range(len(nums)):
            n = nums.pop(0)
            perms = self.permute(nums)
            for perm in perms:
                perm.append(n)
            res.extend(perms)
            nums.append(n)
        return res
```

### 49. Group Anagrams

#### hash table

#### line: 7

The question asked to use divide and conquer
return true.

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            res[tuple(count)].append(s)
        return res.values()
```

### 50. Pow(x, n)

#### D & C

#### line: 10

The question asked to use divide and conquer
return true.

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def helper(x, n):
            if x == 0:
                return 0
            if n == 0:
                return 1
            res = helper(x, n // 2)
            res *= res
            return res * x if n % 2 else res
        res = helper(x, abs(n))
        return res if n >= 0 else 1 / res
```

### 54. Spiral Matrix

#### D & C

#### line: 21

The question asked to use divide and conquer
return true.

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res, left, right, top, bottom = [], 0, len(matrix[0]), 0, len(matrix)
        while left < right and top < bottom:
            # top
            for c in range(left, right):
                res.append(matrix[top][c])
            top += 1
            if top == bottom: break
            # right
            for r in range(top, bottom):
                res.append(matrix[r][right - 1])
            right -= 1
            if left == right: break
            # bottom
            for c in range(right - 1, left - 1, -1):
                res.append(matrix[bottom - 1][c])
            bottom -= 1
            # left
            for r in range(bottom - 1, top - 1, -1):
                res.append(matrix[r][left])
            left += 1
        return res
```

### 59. Spiral Matrix II

#### D & C

#### line: 30

The question asked to use matrix

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        res, count = [[0] * n for i in range(n)], 1
        left, right, top, bottom = 0, n, 0, n
        while left < right and top < bottom:
            # top
            for c in range(left, right):
                res[top][c] = count
                count += 1
            top += 1
            if top == bottom: break
            # right
            for r in range(top, bottom):
                res[r][right - 1] = count
                count += 1
            right -= 1
            if left == right: break
            # bottom
            for c in range(right - 1, left - 1, -1):
                res[bottom - 1][c] = count
                count += 1
            bottom -= 1
            # left
            for r in range(bottom - 1, top - 1, -1):
                res[r][left] = count
                count += 1
            left += 1
        return res
```

### 149. Max Points on a Line

#### D & C

#### line: 28

The question asked to use matrix

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def calcSlope(a, b): # avoid 0 of dx or dy
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            divisor = gcd(dx, dy)
            return (dx / divisor, dy / divisor) # use tuple as key

        if len(points) <= 2:
            return len(points)
        res = 0
        for i in range(len(points)):
            slopes, dups = {}, 1 # [0, 0], [0, 0] or [0, 0], [1, 1], [2, 2]
            for j in range(i + 1, len(points)):
                if points[i] == points[j]:
                    dups += 1
                else:
                    slope = calcSlope(points[i], points[j])
                    if slope in slopes:
                        slopes[slope] += 1
                    else:
                        slopes[slope] = 1
            for slope in slopes:
                res = max(res, slopes[slope] + dups)
        return res
```

### 187. Repeated DNA Sequences

#### set

#### line: 7

The question asked to use matrix

```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        seen, res = set(), set()
        for i in range(len(s) - 9):
            cur = s[i: i + 10]
            if cur in seen:
                res.add(cur)
            seen.add(cur)
        return list(res)
```

### 384. Shuffle an Array

#### Fisher Yates Algorithms

#### line: 7

The question asked to use matrix

```python
class Solution:

    def __init__(self, nums: List[int]):
        self.original = nums[: ]

    def reset(self) -> List[int]:
        return self.original

    def shuffle(self) -> List[int]:
        res = self.original[: ]
        # Fisher Yates Algorithms
        lastIndex = len(res) - 1
        while lastIndex > 0:
            randomIndex = random.randint(0, lastIndex)
            res[lastIndex], res[randomIndex] = res[randomIndex], res[lastIndex]
            lastIndex -= 1
        return res


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()
```

### 380. Insert Delete GetRandom O(1)

#### Fisher Yates Algorithms

#### line: 7

The question asked to use matrix

```python
class RandomizedSet:

    def __init__(self):
        self.set = set()

    def insert(self, val: int) -> bool:
        if val in self.set:
            return False
        self.set.add(val)
        return True

    def remove(self, val: int) -> bool:
        if val in self.set:
            self.set.remove(val)
            return True
        return False


    def getRandom(self) -> int:
        lst = list(self.set)
        random.shuffle(lst)
        return lst[0]


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

### 173. Binary Search Tree Iterator

#### stack

#### line: 10

use stack + dfs iterative solution

```python
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        res = self.stack.pop()
        cur = res.right
        while cur:
            self.stack.append(cur)
            cur = cur.left
        return res.val

    def hasNext(self) -> bool:
        return self.stack != []

```

### 373. Find K Pairs with Smallest Sums

#### D & C

#### line: 10

Keep all possible pairs in the heap and choose k smallest pairs

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        hq, res = [], []
        for i in range(min(len(nums1), k)):
            heapq.heappush(hq, (nums1[i] + nums2[0], nums1[i], nums2[0], 0)) # idx for nums2
        while k > 0 and hq:
            _, n1, n2, idx = heapq.heappop(hq)
            res.append([n1, n2])
            if idx + 1 < len(nums2):
                heapq.heappush(hq, (n1+nums2[idx+1], n1, nums2[idx+1], idx+1))
            k -= 1
        return res
```

### 491. Increasing Subsequences

#### memo

#### line: 12

The question asked to use set + backTracking

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        def backTracking(start, cur):
            if len(cur) > 1:
                res.add(tuple(cur))
            last = cur[-1] if cur else float('-inf')
            for i in range(start, n):
                if nums[i] >= last:
                    cur.append(nums[i])
                    backTracking(i + 1, cur)
                    cur.pop()

        res, n = set(), len(nums)
        backTracking(0, [])
        return res
```

### 743. Network Delay Time

#### min heap

#### line: 14

Use min heap and graph of Dijkstra's algorithms

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        edges = collections.defaultdict(list)
        for u, v, w in times:
            edges[u].append((v, w))

        minHeap, visit, t = [(0, k)], set(), 0
        while minHeap:
            w1, n1 = heapq.heappop(minHeap)
            if n1 in visit:
                continue
            visit.add(n1)
            t = max(t, w1)
            for n2, w2 in edges[n1]:
                if n2 not in visit:
                    heapq.heappush(minHeap, (w1 + w2, n2))
        return t if len(visit) == n else -1
```

### 787. Cheapest Flights Within K Stops

#### min heap

#### line: 11

Use min heap and graph of Dijkstra's algorithms

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        prices = [float('inf')] * n
        prices[src] = 0
        for i in range(k + 1):
            temp = prices.copy()

            for s, d, p in flights:
                if prices[s] == float('inf'):
                    continue
                if prices[s] + p < temp[d]:
                    temp[d] = prices[s] + p
            prices = temp
        return -1 if prices[dst] == float('inf') else prices[dst]
```

### 827. Making A Large Island

#### dfs

#### line: 26

The question asked to use map and dfs

```python
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        def dfs(i, j, color): # color all components
            grid[i][j], area, points = color, 1, [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            for r, c in points:
                if 0 <= r < n and 0 <= c < n and grid[r][c] == 1:
                    area += dfs(r, c, color)
            return area

        n, color, water, island = len(grid), 2, set(), {} # put color, area in island map
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    water.add((i,j))
                elif grid[i][j] == 1:
                    area = dfs(i, j, color)
                    island[color] = area
                    color += 1

        res = island[2] if len(island) > 0 else 0 # is no water, area is island[2], only 1 color
        for i, j in water:
            area, visit, points = 1, set(), [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            for r, c in points:
                if 0 <= r < n and 0 <= c < n:
                    color = grid[r][c]
                    if color not in visit and color in island:
                        area += island[color]
                        visit.add(color)
            res = max(res, area)

        return res
```

### 1387. Sort Integers by The Power Value

#### memo

#### line: 21

The question asked to use divide and conquer
return true.

```python
class Solution:
    def getKth(self, lo: int, hi: int, k: int) -> int:
        memo = {1: 1}
        def helper(n):
            if not n in memo:
                if n % 2:
                    memo[n] = helper(3 * n + 1) + 1
                else:
                    memo[n] = helper(n / 2) + 1
            return memo[n]
        values = [[helper(i), i] for i in range(lo, hi + 1)]
        return sorted(values)[k - 1][1]
```

### 936. Stamping The Sequence

#### greedy

#### line: 24

The greedy method is to stamp the last one first, then the second last, we used '_'
to control every step should be ok, the first stamp must be a full width of stamp, then we have '_'
all other steps should be ok. If not ok, return false.

```python
class Solution:
    def movesToStamp(self, stamp: str, target: str) -> List[int]:
        def canStamp(start):
            flag = False
            for i in range(s):
                if targetList[start + i] == '?':
                    continue
                flag = True
                if stamp[i] != targetList[start + i]:
                    return False
            return flag

        s, t, targetList = len(stamp), len(target), list(target)
        remain, res = t, []
        while remain:
            flag = False

            for i in range(t - s + 1):
                if canStamp(i):
                    flag = True
                    res.append(i)
                    for j in range(s):
                        if targetList[i + j] != '?':
                            targetList[i + j] = '?'
                            remain -= 1
            if not flag:
                return []
        return res[::-1]
```

### 684. Redundant Connection

#### greedy

#### line: 22

The find is to find parent, the union is to union child and parent

```python
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        par = [i for i in range(len(edges) + 1)]
        rank = [1] * (len(edges) + 1)

        def find(n):
            p = par[n]
            while p != par[p]:
                par[p] = par[par[p]] # quick find, can delete this line
                p = par[p]
            return p

        def union(n1, n2):
            p1, p2 = find(n1), find(n2)
            if p1 == p2:
                return False
            if rank[p1] >= rank[p2]:
                par[p2] = p1
                rank[p1] += rank[p2]
            else:
                par[p1] = p2
                rank[p2] += rank[p1]
            return True

        for n1, n2 in edges:
            if not union(n1, n2):
                return [n1, n2]
```

### 326. Power of Three

#### math

#### line: 1

```python
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        return n > 0 and (3**19) % n == 0
```

### 231. Power of Two

#### greedy

#### line: 22

The find is to find parent, the union is to union child and parent

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and 2**round(log(n, 2)) == n
```

### 342. Power of Four

#### greedy

#### line: 22

The find is to find parent, the union is to union child and parent

```python
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and not (n & (n - 1)) and not (n & 0xaaaaaaaa)
```

### 638. Shopping Offers

#### dp: knapsack unbounded

#### line: 12

This is a knapsack problem

```python
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        memo = {}
        def helper(price, special, needs):
            if tuple(needs) in memo:
                return memo[tuple(needs)]
            minPrice = sum([needs[i] * price[i] for i in range(len(needs))])
            for offer in special:
                if all([offer[i] <= needs[i] for i in range(len(needs))]):
                    newNeeds = [needs[i] - offer[i] for i in range(len(needs))]
                    minPrice = min(minPrice, offer[-1] + helper(price, special, newNeeds))
            memo[tuple(needs)] = minPrice
            return minPrice
        return helper(price, special, needs)

```

### 806. Number of Lines To Write String

#### dp: knapsack unbounded

#### line: 9

This is a one loop

```python
class Solution:
    def numberOfLines(self, widths: List[int], s: str) -> List[int]:
        line, pixels, letters, d = 1, 0, "abcdefghijklmnopqrstuvwxyz", {}
        for i, letter in enumerate(letters):
            d[letter] = widths[i]
        for c in s:
            pixels += d[c]
            if pixels > 100:
                line += 1
                pixels = d[c]
        return [line, pixels]
```

######################################################################################################
graph
######################################################################################################

### 1971. Find if Path Exists in Graph

#### graph + bfs

#### line: 16

bfs is used with a queue and visit set, in python it's deque(), bfs can use iterative easily

```python
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        queue, visited = deque(), set()
        queue.append(source)
        visited.add(source)
        while queue:
            node = queue.popleft()
            if node == destination:
                return True
            for adjacent_node in graph[node]:
                if adjacent_node not in visited:
                    queue.append(adjacent_node)
                    visited.add(adjacent_node)
        return False
```

### 1791. Find Center of Star Graph

#### graph

#### line: 7

adjacency list

```python
class Solution:
    def findCenter(self, edges: List[List[int]]) -> int:
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
            for node in graph.keys():
                if len(graph[node]) > 1:
                    return node
```

### 997. Find the Town Judge

#### graph

#### line: 8

degree of graph

```python
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        count = [0] * (n + 1) # do not calc using 0 index
        for u, v in trust: #indegree and outdegree
            count[u] -= 1
            count[v] += 1
        for i in range(1, len(count)):
            if count[i] == n - 1:
                return i
        return -1
```

### 797. All Paths From Source to Target

#### graph + dfs

#### line: 8

dfs of graph

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        def dfs(start, path, res):
            if start == end:
                res.append(path)
            for adjacent_node in graph[start]:
                dfs(adjacent_node, path + [adjacent_node], res)

        end, res = len(graph) - 1, []
        dfs(0, [0], res)
        return res
```

### 1557. Minimum Number of Vertices to Reach All Nodes

#### graph + indegree

#### line: 7

```python
class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        res, count = [], [0] * n # count indegrees
        for u, v in edges:
            count[v] += 1
        for i in range(len(count)): # check indegree of 0
            if count[i] == 0:
                res.append(i)
        return res
```

### 934. Shortest Bridge

```python
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        N, direct, visit = len(grid), [[0, 1], [0, -1], [1, 0], [-1, 0]], set()

        def outOfBounds(r, c):
            return r < 0 or c < 0 or r == N or c == N

        def dfs(r, c):
            if outOfBounds(r, c) or not grid[r][c] or (r, c) in visit:
                return
            visit.add((r, c))
            for dr, dc in direct:
                dfs(r + dr, c + dc)

        def bfs():
            res, q = 0, deque(visit)
            while q:
                for i in range(len(q)):
                    r, c = q.popleft()
                    for dr, dc in direct:
                        row, col = r + dr, c + dc
                        if outOfBounds(row, col) or (row, col) in visit:
                            continue
                        if grid[row][col]:
                            return res
                        q.append((row, col))
                        visit.add((row, col))
                res += 1

        for r in range(N):
            for c in range(N):
                if grid[r][c]:
                    dfs(r, c)
                    return bfs()
```

### 802. Find Eventual Safe States

```python
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        n, safe = len(graph), {}

        def dfs(i):
            if i in safe:
                return safe[i]
            safe[i] = False
            for nei in graph[i]:
                if not dfs(nei):
                    return safe[i]
            safe[i] = True
            return safe[i]
        res = []
        for i in range(n):
            if dfs(i):
                res.append(i)
        return res
```

933. Number of Recent Calls

```python
class RecentCounter:

    def __init__(self):
        self.q = deque()

    def ping(self, t: int) -> int:
        self.q.append(t)
        while self.q[0] < t - 3000:
            self.q.popleft()
        return len(self.q)
```

950. Reveal Cards In Increasing Order

the idea is how to build an arr by using queue, starting from the end to the begining

```python
class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        q = deque()
        deck.sort(reverse = True)
        for card in deck:
            if q:
                q.appendleft(q.pop())
            q.appendleft(card)
        return q
```

1823. Find the Winner of the Circular Game

```python
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        res, q = 0, deque([i for i in range(1, n + 1)])
        while q:
            for i in range(1, k):
                q.append(q.popleft())
            res = q.popleft()
        return res
```

1352. Product of the Last K Numbers

```python
class ProductOfNumbers:

    def __init__(self):
        self.a = [1]

    def add(self, num: int) -> None:
        if not num:
            self.a = [1]
        else:
            self.a.append(self.a[-1] * num)

    def getProduct(self, k: int) -> int:
        if len(self.a) <= k:
            return 0
        return self.a[-1] // self.a[-k - 1]


# Your ProductOfNumbers object will be instantiated and called as such:
# obj = ProductOfNumbers()
# obj.add(num)
# param_2 = obj.getProduct(k)
```

#####################################

2 pointers

443. String Compression

```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        walker, runner = 0, 0
        while runner < len(chars):
            chars[walker] = chars[runner]
            count = 1

            while runner + 1 < len(chars) and chars[runner] == chars[runner+1]:
                runner += 1
                count += 1

            if count > 1:
                for c in str(count):
                    chars[walker+1] = c
                    walker += 1
            runner += 1
            walker += 1
        return walker
```

969. Pancake Sorting

```python
class Solution:
    def pancakeSort(self, arr: List[int]) -> List[int]:
        res, r = [], len(arr)
        while r > 1:
            l = arr.index(max(arr[:r])) #find index of max value
            arr[: l + 1] = reversed(arr[: l + 1]) # reverse to the max value
            res.append(l + 1) #append size
            arr[: r]=reversed(arr[: r]) # reverse all
            res.append(r) # append size
            r -= 1
        return res
```

1382. Balance a Binary Search Tree

```python
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        def inOrder(root, nodes):
            if root:
                inOrder(root.left, nodes)
                nodes.append(root)
                inOrder(root.right, nodes)
            return nodes
        def arrToBST(l, r, nodes):
            if l > r:
                return None
            m = (l + r) // 2
            root = nodes[m]
            root.left = arrToBST(l, m - 1, nodes)
            root.right = arrToBST(m + 1, r, nodes)
            return root
        nodes = inOrder(root, [])
        return arrToBST(0, len(nodes) - 1, nodes)
```

1561. Maximum Number of Coins You Can Get

```python
class Solution:
    def maxCoins(self, piles: List[int]) -> int:
        res, m = 0, len(piles) // 3 # m is the number of groups
        piles.sort(reverse = True) # sort in reverse order
        for i in range(1, 2 * m, 2): # always choose the second large value, ignore third value
            res += piles[i]
        return res
```

### 861. Score After Flipping Matrix

```python
class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        for r in range(rows):
            if grid[r][0] == 0:
                for c in range(cols):
                    grid[r][c] ^= 1

        for c in range(cols):
            cnt = sum(grid[r][c] for r in range(rows))
            if cnt < rows - cnt:
                for r in range(rows):
                    grid[r][c] ^= 1

        return sum(int("".join(map(str, grid[i])), 2) for i in range(rows))
```

### 419. Battleships in a Board

similar to 200 in dfs
O(n) space

```python
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        rows, cols, directions = len(board), len(board[0]), [[-1, 0], [1, 0], [0, 1], [0, -1]]
        def dfs(i, j):
            # board[i][j] == '.' # no use for this line
            for r, c in directions:
                dr = r + i
                dc = c + j
                if dr >= 0 and dc >= 0 and dr < rows and dc < cols and board[dr][dc] == 'X' and (dr, dc) not in visit:
                    visit.add((dr, dc))
                    dfs(dr, dc)

        res, visit = 0, set()
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == 'X' and (i, j) not in visit:
                    dfs(i, j)
                    res += 1
        return res
```

O(1) space

```python
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        res, rows, cols = 0, len(board), len(board[0])
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'X' and (r == 0 or board[r-1][c] == '.') and (c == 0 or board[r][c-1] == '.'):
                    res += 1
        return res
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

### 386. Lexicographical Numbers

```python
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        res = [str(i) for i in range(1, n + 1)]
        res.sort()
        return [int(s) for s in res]
```

```python
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        def dfs(x):
            """Pre-order traverse the tree."""
            if x <= n:
                ans.append(x)
                for i in range(10):
                    dfs(10 * x + i)

        ans = []
        for i in range(1, 10):
            dfs(i)
        return ans
```

### 993. Cousins in Binary Tree

this is much easier than dfs

```python
class Solution:
    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        d = {}
        def dfs(node, parent, depth):
            if not node:
                return
            dfs(node.left, node, depth + 1)
            dfs(node.right, node, depth + 1)
            if node.val in (x, y):
                d[node.val] = (parent, depth)

        dfs(root, None, 0)
        res = x in d and y in d and d[x][1] == d[y][1] and d[x][0] != d[y][0]
        return res
```

##################################################################

### dp

#### 1d array: 70, 118, 119, 121, 338, 392, 509, 746, 1137

###### 118 == 119, 70 == 509 == 1137, 392: 2 pointers

### 70. Climbing Stairs

### line: 5

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        dp = [1]
        for i in range(1, rowIndex + 1):
            addZero = [0] + dp + [0]
            dp = [addZero[i - 1] + addZero[i] for i in range(1, len(addZero))]
        return dp
```

### 118. Pascal's Triangle

### line: 6

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        for i in range(2, numRows + 1):
            addZero = [0] + res[-1] + [0]
            dp = [addZero[i - 1] + addZero[i] for i in range(1, len(addZero))]
            res.append(dp)
        return res
```

### 119. Pascal's Triangle II

### line: 5

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        dp = [1]
        for i in range(1, rowIndex + 1):
            addZero = [0] + dp + [0]
            dp = [addZero[i - 1] + addZero[i] for i in range(1, len(addZero))]
        return dp
```

### 121. Best Time to Buy and Sell Stock

### line: 5

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowest, profit = prices[0], 0
        for price in prices:
            lowest = min(lowest, price)
            profit = max(profit, price - lowest)
        return profit
```

### 338. Counting Bits

### line: 7

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        for i in range(1, len(dp)):
            if i % 2 == 1:
                dp[i] = dp[i // 2] + 1
            else:
                dp[i] = dp[i // 2]
        return dp
```

### 392. Is Subsequence

### 2 pointers

### line: 8

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

### 509. Fibonacci Number

### line: 6

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

### 746. Min Cost Climbing Stairs

### line: 3

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        for i in range(2, len(cost)):
            cost[i] += min(cost[i - 2], cost[i - 1])
        return min(cost[-2:])

```

### 1137. N-th Tribonacci Number

### line: 8

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        if n < 2:
            return n
        elif n == 2:
            return 1
        first, second, third = 0, 1, 1
        for i in range(3, n + 1):
            third, second, first = third + second + first, third, second
        return third
```

### 1646. Get Maximum in Generated Array

### line: 9

```python
class Solution:
    def getMaximumGenerated(self, n: int) -> int:
        if n==0:
            return 0
        dp = [0, 1]
        for i in range(2, n+1):
            if i % 2==0:
                dp.append(dp[i // 2])
            else:
                dp.append(dp[i // 2] + dp[i // 2 + 1])
        return max(dp)
```

### 1025. Divisor Game

```python
class Solution:
    def divisorGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            for j in range(1, i // 2 + 1):
                if i % j == 0 and not dp[i - j]:
                    dp[i] = True # if i need to win, next step to Bob is dp[i - j] must be false
                    break # this is optimal
        return dp[-1]
```

### 198. House Robber

#### line: 4

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        nums = [0] + nums # use nums as dp
        for i in range(2, len(nums)):
            nums[i] = max(nums[i - 1], nums[i - 2] + nums[i])
        return nums[-1]
```

### 213. House Robber II

#### line: 5

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        nums1, nums2 = [0] + nums[: -1], [0] + nums[1: ]
        for i in range(2, len(nums1)):
            nums1[i] = max(nums1[i - 1], nums1[i - 2] + nums1[i])
            nums2[i] = max(nums2[i - 1], nums2[i - 2] + nums2[i])
        return max(nums1[-1], nums2[-1], nums[0])
```

### 740. Delete and Earn

#### line: 6

```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        dp = [0] * (max(nums) + 1) # dp index is houses 0, 1, 2, ...
        for n in nums: # the problem is now house robbers
            dp[n] += n
        for i in range(2, len(dp)):
            dp[i] = max(dp[i - 1], dp[i] + dp[i - 2])
        return dp[-1]
```

### 55. Jump Game

#### line: 7

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        res = 0
        for i in range(len(nums)):
            res = max(res, i + nums[i])
            if (res >= len(nums) - 1):
                return True
            elif res <= i:
                return False
```

### 45. Jump Game II

### dp

```python
# method 1: dp
        dp = [0] * len(nums) # last number of dp is the value
        for i in range(1, len(nums)):
            min_value = float('inf')
            for j in range(i):
                if i - j <= nums[j]:
                    min_value = min(min_value, dp[j] + 1)
            dp[i] = min_value
        return dp[-1] # time too long
```

### greedy

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        res, l, r = 0, 0, 0 # window of indexes
        while r < len(nums) - 1:
            farthest = 0
            for i in range(l, r + 1):
                farthest = max(farthest, i + nums[i])
            l = r + 1
            r = farthest
            res += 1
        return res
```

### 1014. Best Sightseeing Pair

```python
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        res, cur = 0, 0
        for i in range(1, len(values)):
            cur = max(cur, values[i - 1] + i - 1)
            res = max(res, cur + values[i] - i)
        return res
```

### 53. Maximum Subarray

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [float('-inf')] * (len(nums) + 1)
        for i in range(1, len(dp)):
            dp[i] = max(dp[i - 1] + nums[i - 1], nums[i - 1])
        return max(dp)
```

### 918. Maximum Sum Circular Subarray

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        def maxSubarraySum(nums):
            dp = [float('-inf')] * (len(nums) + 1)
            for i in range(1, len(dp)):
                dp[i] = max(dp[i - 1] + nums[i - 1], nums[i - 1])
            return max(dp)

        if len(nums) == 1:
            return nums[0]
        drop = maxSubarraySum(nums[1:])
        pick = sum(nums) - min(0, -maxSubarraySum([-number for number in nums[1:]]))
        return max(drop, pick)
```

### 729. My Calendar I

```python
class TreeNode():
    def __init__(self, s, e):
        self.s = s
        self.e = e
        self.left = None
        self.right = None

class MyCalendar:

    def __init__(self):
        self.root = None

    def book(self, start: int, end: int) -> bool:
        if not self.root:
            self.root = TreeNode(start, end)
            return True
        else:
            return self.insert(start, end, self.root)

    def insert(self, s, e, node):
        if s >= node.e:
            if node.right:
                return self.insert(s, e, node.right)
            else:
                node.right = TreeNode(s, e)
                return True
        elif e <= node.s:
            if node.left:
                return self.insert(s, e, node.left)
            else:
                node.left = TreeNode(s, e)
                return True
        else:
            return False


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)
```

### 731. My Calendar II

```python
class MyCalendarTwo:

    def __init__(self):
        self.lst = []

    def book(self, start: int, end: int) -> bool:
        self.lst.append((start, +1)) # one booking added at startingtime = start
        self.lst.append((end, -1)) # booking ends at timestamp = end

        self.lst.sort()
        overlaps = 0
        for book in self.lst:
            overlaps += book[1]
            if overlaps > 2:
                self.lst.remove((start, +1)) # remove the booking that is causing problems (triple overlap)
                self.lst.remove((end, -1))
                return False
        return True


# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)
```

############################################################################

## dp

### Minimum (Maximum) Path to Reach a Target

### 64. Minimum Path Sum

how to reach the target, check the last step, then find the function

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        rows, cols, first = len(grid), len(grid[0]), grid[0][0]
        # state
        dp = [[first] * cols for i in range(rows)]
        # init rows
        for j in range(1, cols):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        # init cols
        for i in range(1, rows):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        # top-down
        for i in range(1, rows):
            for j in range(1, cols):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[-1][-1]
```

### 120. Triangle

how to reach the target, check the last step, then find the function

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # state
        dp = triangle[-1]
        # bottom-up
        for row in triangle[::-1][1:]:
            for i, v in enumerate(row):
                dp[i] = v + min(dp[i], dp[i + 1])
        return dp[0]
```

### 174. Dungeon Game

```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        rows, cols = len(dungeon), len(dungeon[0])
        dp = [[0] * cols for i in range(rows)]

        # state: need at least 1 point to survive
        dp[-1][-1] = max(1, 1 - dungeon[-1][-1])

        # rows
        for i in range(rows - 2, -1, -1):
            dp[i][-1] = max(1, dp[i + 1][-1] - dungeon[i][-1])
        # cols
        for j in range(cols - 2, -1, -1):
            dp[-1][j] = max(1, dp[-1][j + 1] - dungeon[-1][j])

        # bottom-up
        for i in range(rows-2, -1, -1):
            for j in range(cols - 2, -1, -1):
                dp[i][j] = max(1, min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j])
        return dp[0][0]
```

### 221. Maximal Square

same as 1277

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        res, rows, cols = 0, len(matrix) + 1, len(matrix[0]) + 1
        # state
        dp = [[0] * cols  for i in range(rows)]
        # top-down
        for i in range(1, rows): # rows
            for j in range(1, cols): # cols
                if matrix[i-1][j-1] == '1':
                    dp[i][j] = min([dp[i-1][j-1], dp[i-1][j], dp[i][j-1]]) + 1
                    res = max(res, dp[i][j])
        return res ** 2
```

### 1277. Count Square Submatrices with All Ones

same as 221

```python
class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        res, rows, cols = 0, len(matrix) + 1, len(matrix[0]) + 1
        dp = [[0] * cols  for i in range(rows)]
        for i in range(1, rows): # rows
            for j in range(1, cols): # cols
                if matrix[i - 1][j - 1] == 1:
                    dp[i][j] = min([dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]]) + 1
        res = sum([sum(item) for item in dp])
        return res
```

### 931. Minimum Falling Path Sum

same as 1289

```python
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        rows, cols = len(matrix), len(matrix[0]) + 2
        dp = [[float('inf')] * cols for i in range(rows)]
        for j in range(1, cols - 1):
            dp[0][j] = matrix[0][j - 1]
        for i in range(1, rows):
            for j in range(1, cols - 1):
                dp[i][j] = min(dp[i - 1][j - 1: j + 2]) + matrix[i][j - 1]
        return min(dp[-1])
```

### 1289. Minimum Falling Path Sum II

same as 931

```python
class Solution:
    def minFallingPathSum(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0]) + 2
        dp = [[float('inf')] * cols for i in range(rows)]
        for j in range(1, cols - 1):
            dp[0][j] = grid[0][j - 1]
        for i in range(1, rows):
            for j in range(1, cols - 1):
                dp[i][j] = min(dp[i - 1][: j] + dp[i - 1][j + 1:]) + grid[i][j - 1]
        return min(dp[-1])
```

### 576. Out of Boundary Paths

```python
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        memo = {}
        def dfs(i, j, maxMove):
            if (i, j, maxMove) in memo:
                return memo[(i, j, maxMove)]
            if maxMove < 0:
                return 0
            if i < 0 or i >= m or j < 0 or j >= n:
                return 1
            res, directions = 0, [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for r, c in directions:
                ans = dfs(r, c, maxMove - 1)
                memo[(r, c, maxMove - 1)] = ans
                res += ans
            return res

        return dfs(startRow, startColumn, maxMove) % (10 ** 9 + 7)
```

### 688. Knight Probability in Chessboard

```python
class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        @functools.lru_cache(None)
        def dfs(r, c, k):
            if r < 0 or r >= n or c < 0 or c >= n:
                return 0
            elif k == 0:
                return 1
            else:
                res, directions = 0, [(r + 2, c + 1), (r + 1, c + 2), (r - 1, c + 2), (r - 2, c + 1),
                             (r - 2, c - 1), (r - 1, c - 2), (r + 1, c - 2), (r + 2, c - 1)]
                for row, col in directions:
                    res += dfs(row, col, k - 1) / 8
                return res

        return dfs(row, column, k)
```

### 669. Trim a Binary Search Tree

### dfs

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if not root:
            return None
        elif root.val < low:
            return self.trimBST(root.right, low, high)
        elif root.val > high:
            return self.trimBST(root.left, low, high)
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root
```

### 1254. Number of Closed Islands

```python
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        def dfs(i, j):
            if grid[i][j] == 1:
                return True
            elif i <= 0 or i >= row - 1 or j <= 0 or j >= col - 1:
                return False
            grid[i][j] = 1
            up = dfs(i - 1, j)
            down = dfs(i + 1, j)
            left = dfs(i, j - 1)
            right = dfs(i, j + 1)
            return up and down and left and right

        res, row, col = 0, len(grid),len(grid[0])
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                if not grid[i][j] and dfs(i, j):
                    res += 1
        return res
```

### 508. Most Frequent Subtree Sum

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node):
            if node is None:
                return 0
            res = node.val
            if node.left:
                res += dfs(node.left)
            if node.right:
                res += dfs(node.right)
            d[res] += 1
            return res

        d = defaultdict(int)
        dfs(root)
        max_frequency = max(d.values())
        return [key for key, val in d.items() if val == max_frequency]
```

### bfs template

https://leetcode.com/problems/count-good-nodes-in-binary-tree/discuss/635351/BFS-With-MaxValue-or-Template-of-Similar-Problems-Python

### 1448. Count Good Nodes in Binary Tree

```python
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        res, q = 0, deque([(root,-inf)])
        while q:
            node, maxval = q.popleft()
            if node.val >= maxval:
                res += 1
            if node.left:
                q.append((node.left, max(maxval, node.val)))
            if node.right:
                q.append((node.right, max(maxval, node.val)))
        return res
```

### 559. Maximum Depth of N-ary Tree

```python
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        res, q = -(math.inf), deque([(root, 1)])
        while q:
            node, depth = q.popleft()
            if not node.children:
                res = max(res, depth)
            for i in node.children:
                q.append((i, depth + 1))
        return res
```

### 112. Path Sum

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        q = deque([(root,0)])
        while q:
            node, value = q.popleft()
            if not node.left and not node.right:
                if targetSum == value + node.val:
                    return True
            if node.left:
                q.append((node.left,value + node.val))
            if node.right:
                q.append((node.right,value + node.val))
        return False
```

### 100. Same Tree

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        q = deque([p, q])
        while q:
            node1, node2= q.popleft(), q.popleft()
            if not node1 and not node2:
                continue
            if not node1 or not node2 or node1.val != node2.val:
                return False
            q.append(node1.left)
            q.append(node2.left)
            q.append(node1.right)
            q.append(node2.right)
        return True
```

### 101. Symmetric Tree

```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        q = deque([root, root])
        while q:
            t1, t2 = q.popleft(), q.popleft()
            if not t1 and not t2:
                continue
            if not t1 or not t2 or t1.val != t2.val:
                return False
            q.append(t1.right)
            q.append(t2.left)
            q.append(t1.left)
            q.append(t2.right)
        return True
```

### 102. Binary Tree Level Order Traversal

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

### 865. Smallest Subtree with all the Deepest Nodes

```python
class Solution:
    def findDepth(self, node):
        if not node:
            return 0
        return 1 + max(self.findDepth(node.left), self.findDepth(node.right))

    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        if self.findDepth(root.left) == self.findDepth(root.right):
            return root
        elif self.findDepth(root.left) > self.findDepth(root.right):
            return self.subtreeWithAllDeepest(root.left)
        else:
            return self.subtreeWithAllDeepest(root.right)
```

### 1123. Lowest Common Ancestor of Deepest Leaves

```python
class Solution:
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def findDepth(node):
            if not node:
                return 0
            return 1 + max(findDepth(node.left), findDepth(node.right))

        if findDepth(root.left) == findDepth(root.right):
            return root
        elif findDepth(root.left) > findDepth(root.right):
            return self.lcaDeepestLeaves(root.left)
        else:
            return self.lcaDeepestLeaves(root.right)
```

## Union Find

### 128. Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        res, s = 0, set(nums)
        while s:
            n = s.pop()
            p, q = n + 1, n - 1
            ans = 1
            while p in s:
                ans += 1
                s.remove(p)
                p += 1
            while q in s:
                ans += 1
                s.remove(q)
                q -= 1
            res = max(res, ans)
        return res
```

### 721. Accounts Merge

```python
class UF:
    def __init__(self, N):
        self.parents = list(range(N))

    def find(self, x):
        if self.parents[x] != x:
            return self.find(self.parents[x])
        return x

    def union(self, child, parent):
        self.parents[self.find(child)] = self.find(parent)

class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        uf, ownership = UF(len(accounts)), {}
        for i, (_, *emails) in enumerate(accounts):
            for email in emails:
                if email in ownership:
                    uf.union(i, ownership[email])
                ownership[email] = i

        ans = collections.defaultdict(list)
        for email, owner in ownership.items():
            ans[uf.find(owner)].append(email)
        return [[accounts[i][0]] + sorted(emails) for i, emails in ans.items()]
```

### 684. Redundant Connection

```python
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        par = [i for i in range(len(edges) + 1)]

        def find(n):
            if par[n] != n:
                return find(par[n])
            return n

        def union(n1, n2):
            par[find(n2)] = find(n1)

        for n1, n2 in edges:
            p1, p2 = find(n1), find(n2)
            if p1 != p2:
                union(n1, n2)
            else:
                return [n1, n2]
```

## 1 stone game

### 877. Stone Game

```python
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        cache = {}
        def dfs(l, r):
            if l > r:
                return 0
            if (l, r) in cache:
                return cache[(l, r)]
            alice = True if (r - l) % 2 else False
            left = piles[l] if alice else 0
            right = piles[r] if alice else 0
            cache[(l, r)] = max(dfs((l + 1), r) + left, dfs(l, (r - 1)) + right)
            return cache[(l, r)]
        return dfs(0, len(piles) - 1) > (sum(piles)) // 2
```

### 1140. Stone Game II

```python
class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        N = len(piles)
        @lru_cache(None)
        def miniMax(idx, M):
            if idx == N:
                return 0
            res = float('-inf')
            for X in range(1, 2 * M + 1):
                stones = sum(piles[idx: idx + X])
                score = stones - miniMax(min(N, idx + X), max(M, X))
                res = max(res, score)
            return res
        return (sum(piles) + miniMax(0, 1)) // 2
```

### 1406. Stone Game III

```python
class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        N, M = len(stoneValue), 3
        @lru_cache(None)
        def miniMax(idx):
            if idx == N:
                return 0
            res, stones = float('-inf'), 0
            for X in range(idx, min(idx + 3, len(stoneValue))):
                stones += stoneValue[X]
                score = stones - miniMax(X + 1)
                res = max(res, score)
            return res
        res = miniMax(0)
        return 'Alice' if res > 0 else 'Bob' if res < 0 else'Tie'
```

### 1510. Stone Game IV

```python
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def dfs(n):
            if not n:
                return False
            for i in range(1, int(sqrt(n)) + 1):
                if not dfs(n - i * i):
                    return True
            return False
        return dfs(n)
```

### 1686. Stone Game VI

```python
class Solution:
    def stoneGameVI(self, aliceValues: List[int], bobValues: List[int]) -> int:
        res, alice, bob = [], 0, 0
        for i in range(len(aliceValues)):
            res.append((aliceValues[i] + bobValues[i], aliceValues[i], bobValues[i]))
        res.sort(reverse = True)

        for i in range(len(res)):
            if i % 2 == 0:
                alice += res[i][1]
            else:
                bob += res[i][2]
        return 1 if alice > bob else -1 if alice < bob else 0
```

## 2 jump game

### 1871. Jump Game VII

```python
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        q, farthest = deque([0]), 0
        while q:
            i = q.popleft()
            start = max(i + minJump, farthest + 1)
            for j in range(start, min(i + maxJump + 1, len(s))):
                if s[j] == '0':
                    if j == len(s) - 1:
                        return True
                    q.append(j)
            farthest = i + maxJump
        return False
```

## 3 stock price
