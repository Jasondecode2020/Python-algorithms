### 1 217. Contains Duplicate

Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

Example 1:

Input: nums = [1,2,3,1]
Output: true
Example 2:

Input: nums = [1,2,3,4]
Output: false

```js
var containsDuplicate = function (nums) {
  const set = new Set();
  for (let item of nums) {
    if (set.has(item)) {
      return true;
    } else {
      set.add(item);
    }
  }
  return false;
};
```

### 2 169. Majority Element

Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

Example 1:

Input: nums = [3,2,3]
Output: 3
Example 2:

Input: nums = [2,2,1,1,1,2,2]
Output: 2

```js
var majorityElement = function (nums) {
  let count = 0;
  let majority = null;
  for (let num of nums) {
    if (count === 0) {
      majority = num;
    }
    if (num === majority) {
      count++;
    } else {
      count--;
    }
  }
  return majority;
};
```

### 3 136. Single Number

Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

Example 1:

Input: nums = [2,2,1]
Output: 1
Example 2:

Input: nums = [4,1,2,1,2]
Output: 4
Example 3:

Input: nums = [1]
Output: 1

```js
var singleNumber = function (nums) {
  const set = new Set();
  for (let num of nums) {
    if (set.has(num)) {
      set.delete(num);
    } else {
      set.add(num);
    }
  }
  return [...set][0];
};
```

### 405. Convert a Number to Hexadecimal

Easy

Given an integer num, return a string representing its hexadecimal representation. For negative integers, two’s complement method is used.

All the letters in the answer string should be lowercase characters, and there should not be any leading zeros in the answer except for the zero itself.

Note: You are not allowed to use any built-in library method to directly solve this problem.

Example 1:

Input: num = 26
Output: "1a"
Example 2:

Input: num = -1
Output: "ffffffff"

Constraints:

-231 <= num <= 231 - 1

```js
var toHex = function (num) {
  var arr = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
  ];
  if (num == 0) return "0";
  if (num < 0) num += Math.pow(2, 32);
  var res = "";
  while (num > 0) {
    var digit = num % 16;
    res = arr[digit] + res;
    num = Math.floor(num / 16);
  }
  return res;
};
```
