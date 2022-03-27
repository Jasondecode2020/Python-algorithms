# Day 1 22/3/26

### 1. Two Sum

3min 2min

```js
var twoSum = function (nums, target) {
  const map = new Map();
  for (let i = 0; i < nums.length; i++) {
    const res = target - nums[i];
    if (map.has(res)) return [map.get(res), i];
    map.set(nums[i], i);
  }
};
```

### 9. Palindrome Number

9min 2min

```js
var isPalindrome = function (x) {
  if (x < 0) return false;
  const reversePositiveNumber = (n) => {
    // template
    let ans = 0;
    while (n) {
      ans = ans * 10 + (n % 10);
      n = parseInt(n / 10);
    }
    return ans;
  };
  return reversePositiveNumber(x) === x;
};
```

### 13. Roman to Integer

7min 6min
e.g. IV VS VI

```js
var romanToInt = function (s) {
  const map = {
    I: 1,
    V: 5,
    X: 10,
    L: 50,
    C: 100,
    D: 500,
    M: 1000,
  };
  let res = 0;
  let prev = map["M"];
  for (const c of s) {
    res += map[c];
    if (map[c] > prev) res -= prev * 2;
    prev = map[c];
  }
  return res;
};
```

### 14. Longest Common Prefix

10min 7min

```js
var longestCommonPrefix = function (strs) {
  let res = "";
  for (let i = 0; i < strs[0].length; i++) {
    const prefix = strs[0].slice(0, i + 1);
    if (strs.every((str) => str.slice(0, i + 1) === prefix)) res = prefix;
    else break;
  }
  return res;
};
```

### 20. Valid Parentheses

```js
var isValid = function (s) {
  const characters = ["(", ")", "{", "}", "[", "]"];
  const stack = [];
  for (let c of s) {
    const lastIndex = stack.length - 1;
    if (
      (stack[lastIndex] === characters[0] && c === characters[1]) ||
      (stack[lastIndex] === characters[2] && c === characters[3]) ||
      (stack[lastIndex] === characters[4] && c === characters[5])
    )
      stack.pop();
    else stack.push(c);
  }
  return stack.length === 0;
};
```

### 21. Merge Two Sorted Lists

```js
var mergeTwoLists = function (list1, list2) {
  const dummy = new ListNode();
  let p = dummy;
  while (list1 && list2) {
    if (list1.val > list2.val) {
      p.next = new ListNode(list2.val);
      list2 = list2.next;
    } else {
      p.next = new ListNode(list1.val);
      list1 = list1.next;
    }
    p = p.next;
  }
  p.next = list1 || list2;
  return dummy.next;
};
```

### 26. Remove Duplicates from Sorted Array

use idx to record the result, let i check if there is new num

```js
var removeDuplicates = function (nums) {
  let idx = 1;
  for (let i = 1; i < nums.length; i++) {
    if (nums[i - 1] != nums[i]) {
      nums[idx] = nums[i];
      idx++;
    }
  }
  return idx;
};
```

### 27. Remove Element

```js
var removeElement = function (nums, val) {
  let l = 0;
  let r = nums.length - 1;
  while (l < r) {
    if (nums[l] === val && nums[r] !== val) {
      [nums[l], nums[r]] = [nums[r], nums[l]];
      l++;
      r--;
    } else if (nums[l] === val && nums[r] === val) {
      r--;
    } else {
      l++;
    }
  }
  if (nums[l] === val) return l;
  return l + 1;
};
```

### 28. Implement strStr()

need todo

```js
var strStr = function (haystack, needle) {
  if (needle === "") return 0;

  // define lps array
  const lps = new Array(needle.length).fill(0);
  let prevLPS = 0;
  let i = 1;
  while (i < needle.length) {
    if (needle[i] === needle[prevLPS]) {
      lps[i] = prevLPS + 1;
      prevLPS += 1;
      i += 1;
    } else if (prevLPS === 0) {
      lps[i] = 0;
      i += 1;
    } else {
      prevLPS = lps[prevLPS - 1];
    }
  }

  // use lps to get res
  i = 0; // ptr for haystack
  let j = 0; // ptr for needle
  while (i < haystack.length) {
    if (haystack[i] === needle[j]) {
      i++;
      j++;
    } else {
      if (j === 0) {
        i++;
      } else {
        j = lps[j - 1];
      }
    }
    if (j === needle.length) return i - needle.length;
  }
  return -1;
};
```

### 35. Search Insert Position

```js
var searchInsert = function (nums, target) {
  let l = 0;
  let r = nums.length - 1;
  while (l <= r) {
    const m = l + Math.floor((r - l) / 2);
    if (nums[m] > target) r = m - 1;
    else if (nums[m] < target) l = m + 1;
    else return m;
  }
  return l;
};
```
