### 1 15. 3Sum

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Example 2:

Input: nums = []
Output: []
Example 3:

Input: nums = [0]
Output: []

```js
var threeSum = function (nums) {
  nums.sort((a, b) => a - b);
  const res = [];
  for (let i = 0; i < nums.length; i++) {
    if (i > 0 && nums[i] === nums[i - 1]) {
      continue; // skip same elements to avoid duplicate triplets
    }
    let start = i + 1,
      end = nums.length - 1;
    while (start < end) {
      const sum = nums[i] + nums[start] + nums[end];
      if (sum === 0) {
        res.push([nums[i], nums[start], nums[end]]);
        start++;
        end--;
        while (start < end && nums[start] === nums[start - 1]) {
          start += 1; // skip same elements to avoid duplicate triplets
        }
        while (start < end && nums[end] === nums[end + 1]) {
          end -= 1; // skip same elements to avoid duplicate triplets
        }
      } else if (sum < 0) {
        start++;
      } else {
        end--;
      }
    }
  }

  return res;
};
```
