### 1 217. Contains Duplicate

keyword: set, s means set

note: set stores unique numbers, we can put all numbers in the set one by one
if there is a duplicate number, just return true. Otherwise put the number in the set.
if there is no duplicate number after putting all numbers inside the set, we can safely
return false.

```js
var containsDuplicate = function (nums) {
  const s = new Set();
  for (const num of nums) {
    if (s.has(num)) return true;
    s.add(num);
  }
  return false;
};
```

### 2 53. Maximum Subarray

```js
// O(n) dp
var maxSubArray = function (nums) {
  // the first number of dp  is nums[0], other numbers are depended on nums[i] and dp[i - 1]
  // this means the maximum subarray can be nums[i] or continuous of nums[i] + dp[i - 1]
  const dp = new Array(nums.length).fill(nums[0]);
  // set res be the first number, this maybe a result
  let res = dp[0];

  for (let i = 1; i < nums.length; i++) {
    dp[i] = Math.max(nums[i], nums[i] + dp[i - 1]);
    res = Math.max(res, dp[i]);
  }

  return res;
};
```

```js
// O(n), S(1) Kadane Algorithm
/*
Some more problems based on Kadane Algorithm :)
121. Best Time to Buy and Sell Stock - https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
152. Maximum Product Subarray - https://leetcode.com/problems/maximum-product-subarray/
918. Maximum Sum Circular Subarray - https://leetcode.com/problems/maximum-sum-circular-subarray/
978. Longest Turbulent Subarray - https://leetcode.com/problems/longest-turbulent-subarray/
1749. Maximum Absolute Sum of Any Subarray - https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/
*/
var maxSubArray = function (nums) {
  let prev = 0,
    res = nums[0]; // prev is max of all numbers before the current number
  for (const num of nums) {
    prev = Math.max(num, num + prev); // the max can be either num or num + prev
    res = Math.max(res, prev);
  }

  return res;
};
```

```js
// divide and conquer
// note: help is function for dfs or recursive function
var maxSubArray = function (nums) {
  const helper = (nums, l, r) => {
    if (l === r) return nums[l];
    const mid = l + Math.floor((r - l) / 2);

    let sum = 0;
    let leftMax = -Infinity;
    for (i = mid; i >= l; i--) {
      sum += nums[i];
      leftMax = Math.max(leftMax, sum);
    }

    sum = 0;
    let rightMax = -Infinity;
    for (j = mid + 1; j <= r; j++) {
      sum += nums[j];
      rightMax = Math.max(rightMax, sum);
    }
    return Math.max(
      helper(nums, l, mid),
      helper(nums, mid + 1, r),
      leftMax + rightMax
    );
  };

  return helper(nums, 0, nums.length - 1);
};
```

### 3 1523. Count Odd Numbers in an Interval Range

```js
var countOdds = function (low, high) {
  const helper = (n) => Math.floor(n / 2);
  let carryHigh = 0;
  if (high % 2 === 1) carryHigh = 1;
  return helper(high) - helper(low) + carryHigh;
};
```

### 4 1491. Average Salary Excluding the Minimum and Maximum Salary

```js
var average = function (salary) {
  let minSalary = Infinity;
  let maxSalary = -Infinity;
  let sumSalary = 0;

  for (const s of salary) {
    minSalary = Math.min(minSalary, s);
    maxSalary = Math.max(maxSalary, s);
    sumSalary += s;
  }

  return (sumSalary - minSalary - maxSalary) / (salary.length - 2);
};
```

### 5 509. Fibonacci Number

```js
var fib = function (n) {
  if (n < 2) return n;
  let first = 0;
  let second = 1;
  for (let i = 2; i < n + 1; i++) {
    [first, second] = [second, second + first]; // there is no sequence for bracket operator
  }
  return second;
};
```

### 6 1137. N-th Tribonacci Number

```js
var tribonacci = function (n) {
  if (n < 2) return n;
  let first = 0;
  let second = 1;
  let third = 1;
  for (let i = 3; i < n + 1; i++)
    [first, second, third] = [second, third, first + second + third];
  return third;
};
```

### 7 1848. Minimum Distance to the Target Element

```js
var getMinDistance = function (nums, target, start) {
  let res = Infinity;
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] === target) {
      res = Math.min(res, Math.abs(i - start));
    }
  }
  return res;
};
```

### 8 1854. Maximum Population Year

```js
var maximumPopulation = function (logs) {
  let dp = new Array(2050 - 1950 + 1).fill(0);
  for (const log of logs) {
    for (let i = log[0]; i < log[1]; i++) {
      dp[i - 1950] += 1;
    }
  }
  const maxPopulation = Math.max(...dp);
  for (let i = 0; i < dp.length; i++) {
    if (dp[i] === maxPopulation) return i + 1950;
  }
};
```
