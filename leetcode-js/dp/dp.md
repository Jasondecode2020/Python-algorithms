## 1. 1 use only 1-d array to solve dp is the easiest case

### 1 509. Fibonacci Number

This is a base case, leetcode 1137 and leetcode 70 used the technique

method 1: recursion, in recursive function, the deep of the function stack
is depended on n, and time complexcity is O(2 ** n) which is really big

```js
var fib = function(n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
};
```

method 2: dp

this is why we use dp O(n) space is O(n)

```js
var fib = function(n) {
    const dp = new Array(n + 1).fill(0);
    dp[1] = 1;
    for (let i = 2; i < n + 1; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
};
```

method 3: dp

this is why we use dp O(n) space is O(1)
```js
var fib = function(n) {
    if (n < 2) return n
    let a = 0
    let b = 1
    for (let i = 2; i < n + 1; i++) [a, b] = [b, a + b]
    return b
};
```

### 2 1137. N-th Tribonacci Number

similar to 509. Fibonacci Number

method 1: dp this is why we use dp O(n) space is O(n)

```js
var tribonacci = function(n) {
    const dp = new Array(n + 1).fill(0);
    dp[1] = 1;
    dp[2] = 1;
    for (let i = 3; i < n + 1; i++) {
        dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3];
    }
    return dp[n];
};
```

method 2: this is why we use dp O(n) space is O(1)

```js
var tribonacci = function(n) {
    if (n < 2) return n
    else if (n === 2) return 1
    let a = 0
    let b = 1
    let c = 1
    for (let i = 3; i < n + 1; i++) [a, b, c] = [b, c, a + b + c]
    return c
};
```

### 3 70. Climbing Stairs

method: dp, same as 509. Fibonacci Number

this is why we use dp O(n) space is O(1)

```js
var climbStairs = function(n) {
    if (n < 3) return n
    let a = 1
    let b = 2
    for (let i = 3; i < n + 1; i++) [a, b] = [b, a + b]
    return b
};
```

### 4 746. Min Cost Climbing Stairs

Input: cost = [10,15,20]
Output: 15

use dp = [10, 15, 30], result is the min of the last 2 numbers.
if cost = [10, 15], result is 10
find a dp array represent min cost to reach each index, return min of the 2
```js
var minCostClimbingStairs = function(cost) {
    const n = cost.length
    const dp = new Array(n).fill(0)
    dp[0] = cost[0]
    dp[1] = cost[1]
    for (let i = 2; i < n; i++) {
        dp[i] = Math.min(dp[i - 1] + cost[i], dp[i - 2] + cost[i])
    }
    return Math.min(dp[n - 2], dp[n - 1])
};
```

### 5 198. House Robber

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

dp : max num up to this point

```js
var rob = function(nums) {
    const n = nums.length + 1
    const dp = new Array(n).fill(0); // dp[0] = 0 is used and designed for this
    dp[1] = nums[0];
    let res = dp[1];
    for (let i = 2; i < n; i++) {
        dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1])
        if (dp[i] > res) res = dp[i];
    }
    return res;
};
```

### 6 55. Jump Game

// need to know is num[idx] === 0 cannot make sure false,
// need to add max <= idx means this point cannot pass to end

```js
var canJump = function(nums) {
    let idx = 0;
    let max = 0;
    let target = nums.length - 1;
    while (idx < nums.length) {
        max = Math.max(max, nums[idx] + idx);
        if (max >= target) return true;
        if (nums[idx] === 0 && max <= idx) return false; 
        idx++;
    }
};
```
### 313. Super Ugly Number

```js
var nthSuperUglyNumber = function(n, primes) {
    // start dp from index 1, multiples also start from index 1
    const multiples = new Array(primes.length).fill(1);
    const dp = new Array(n + 1).fill(1);
    for (let i = 2; i < n + 1; i++) {
        dp[i] = Infinity;
        let currIndex = [];
        for (let j = 0; j < primes.length; j++) {
            const minValue = dp[multiples[j]] * primes[j];
            if (minValue < dp[i] && minValue > dp[i - 1]) {
                dp[i] = minValue;
                currIndex = [j];
            } else if (minValue === dp[i]) {
                currIndex.push(j);
            }
        }
        // increase of all indice if have the same value
        for (let i = 0; i < currIndex.length; i++) {
            multiples[currIndex[i]]++;
        }
    }
    return dp[n];
};
```

300. Longest Increasing Subsequence

```js
var lengthOfLIS = function(nums) {
    const dp = new Array(nums.length).fill(1);
    for (let i = 1; i < nums.length; i++) {
        for (let j = i - 1; j >= 0; j--) {
            if (nums[i] > nums[j]) dp[i] = Math.max(dp[i], dp[j] + 1);
        }
    }
    return Math.max(...dp);
};
```