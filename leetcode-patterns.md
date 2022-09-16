### 1

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

### 2

268. Missing Number

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n, s = len(nums), sum(nums)
        return n * (n + 1) // 2 - s
```

### 3

448. Find All Numbers Disappeared in an Array

```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        res = []
        for n in nums:
            nums[abs(n) - 1] = -abs(nums[abs(n) - 1])
        for i, v in enumerate(nums):
            if v > 0:
                res.append(i + 1)
        return res
```

### 4

136. Single Number

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for n in nums:
            res ^= n
        return res
```

### 5

70. Climbing Stairs

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 2:
            return n
        first, second = 1, 2
        for i in range(3, n + 1):
            second, first = first + second, second
        return second
```

### 6

121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowest, profit = prices[0], 0
        for price in prices:
            lowest = min(lowest, price)
            profit = max(profit, price - lowest)
        return profit
```

### 7

303. Range Sum Query - Immutable

```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.nums = nums
        for i in range(1, len(self.nums)):
            self.nums[i] += self.nums[i - 1]

    def sumRange(self, left: int, right: int) -> int:
        leftValue = 0 if left - 1 < 0 else self.nums[left - 1]
        return self.nums[right] - leftValue
```

### 8

338. Counting Bits

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            if i % 2 == 0:
                dp[i] = dp[i // 2]
            else:
                dp[i] = dp[i // 2] + 1
        return dp
```

### 9

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

### 10

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
