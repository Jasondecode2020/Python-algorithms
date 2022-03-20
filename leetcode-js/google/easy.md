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
// O(n)
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
