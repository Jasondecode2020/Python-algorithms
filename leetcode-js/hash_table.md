### 1 1. Two Sum

<Easy>

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.

#### method: 1

```js
var twoSum = function (nums, target) {
  const map = new Map();
  for (let i = 0; i < nums.length; i++) {
    const res = target - nums[i];
    if (map.has(res)) {
      return [map.get(res), i];
    } else {
      map.set(nums[i], i);
    }
  }
};
```

1. new Map()
2. map.has(res)
3. map.get(res)
4. map.set(nums[i], i)

#### method: 2

```js
var twoSum = function (nums, target) {
  const map = {};
  for (let i = 0; i < nums.length; i++) {
    const res = target - nums[i];
    if (map[res] || map[res] === 0) {
      return [map[res], i];
    } else {
      map[nums[i]] = i;
    }
  }
};
```

Why we have to use new Map(), if not may have bugs like neglect map[res] === 0

### 2 3. Longest Substring Without Repeating Characters

<Medium>

Given a string s, find the length of the longest substring without repeating characters.

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
Example 4:

Input: s = ""
Output: 0

Constraints:

0 <= s.length <= 5 \* 104
s consists of English letters, digits, symbols and spaces.

```js
var lengthOfLongestSubstring = function (s) {
  const map = new Map();
  let longest = 0;
  let left = 0;
  for (let right = 0; right < s.length; right++) {
    if (map.has(s[right])) {
      left = Math.max(map.get(s[right]) + 1, left);
    }
    longest = Math.max(longest, right - left + 1);
    map.set(s[right], right);
  }
  return longest;
};
```

Write the code for best solution

### 2 1941. Check if All Characters Have Equal Number of Occurrences

<Easy>

Given a string s, return true if s is a good string, or false otherwise.

A string s is good if all the characters that appear in s have the same number of occurrences (i.e., the same frequency).

Example 1:

Input: s = "abacbc"
Output: true
Explanation: The characters that appear in s are 'a', 'b', and 'c'. All characters occur 2 times in s.

Example 2:

Input: s = "aaabb"
Output: false
Explanation: The characters that appear in s are 'a' and 'b'.
'a' occurs 3 times while 'b' occurs 2 times, which is not the same number of times.

Constraints:

1 <= s.length <= 1000
s consists of lowercase English letters.

#### method 1: too complicated

used array, set and map

```js
var areOccurrencesEqual = function (s) {
  let map = new Map();
  for (let i = 0; i < s.length; i++) {
    if (map.has(s[i])) {
      let num = map.get(s[i]);
      map.set(s[i], ++num);
    } else {
      map.set(s[i], 1);
    }
  }
  let res = [];
  for (const value of map.values()) {
    res.push(value);
  }
  return new Set(res).size === 1;
};
```

#### method 2: simple way

```js
var areOccurrencesEqual = function (s) {
  let map = {};
  for (let c of s) {
    map[c] = (map[c] || 0) + 1;
  }
  const firstValue = map[s[0]];
  for (let c in map) {
    if (map[c] != firstValue) return false;
  }
  return true;
};
```

#### method 3: right way

```js
var areOccurrencesEqual = function (s) {
  const map = new Map();
  for (let c of s) {
    if (map.has(c)) {
      map.set(c, map.get(c) + 1);
    } else {
      map.set(c, 1);
    }
  }
  const firstValue = map.get(s[0]);
  for (let c of map.values()) {
    if (c !== firstValue) return false;
  }
  return true;
};
```

#### method 4: use the right var

```js
var areOccurrencesEqual = function (s) {
  const map = new Map();
  for (let c of s) {
    if (map.has(c)) {
      map.set(c, map.get(c) + 1);
    } else {
      map.set(c, 1);
    }
  }
  const firstValue = map.get(s[0]);
  for (let val of map.values()) {
    if (val !== firstValue) return false;
  }
  return true;
};
```

For hashmap problem, we should use new Map() t0 avoid bugs

### 2 1935. Maximum Number of Words You Can Type

Easy

There is a malfunctioning keyboard where some letter keys do not work. All other keys on the keyboard work properly.

Given a string text of words separated by a single space (no leading or trailing spaces) and a string brokenLetters of all distinct letter keys that are broken, return the number of words in text you can fully type using this keyboard.

Example 1:

Input: text = "hello world", brokenLetters = "ad"
Output: 1
Explanation: We cannot type "world" because the 'd' key is broken.
Example 2:

Input: text = "leet code", brokenLetters = "lt"
Output: 1
Explanation: We cannot type "leet" because the 'l' and 't' keys are broken.
Example 3:

Input: text = "leet code", brokenLetters = "e"
Output: 0
Explanation: We cannot type either word because the 'e' key is broken.

Constraints:

1 <= text.length <= 104
0 <= brokenLetters.length <= 26
text consists of words separated by a single space without any leading or trailing spaces.
Each word only consists of lowercase English letters.
brokenLetters consists of distinct lowercase English letters.

```js
var canBeTypedWords = function (text, brokenLetters) {
  const textArray = text.split(" ");
  let count = textArray.length;
  for (word of textArray) {
    for (let c of brokenLetters) {
      if (word.includes(c)) {
        count--;
        break;
      }
    }
  }
  return count;
};
```

### 3 1893. Check if All the Integers in a Range Are Covered

Easy

You are given a 2D integer array ranges and two integers left and right. Each ranges[i] = [starti, endi] represents an inclusive interval between starti and endi.

Return true if each integer in the inclusive range [left, right] is covered by at least one interval in ranges. Return false otherwise.

An integer x is covered by an interval ranges[i] = [starti, endi] if starti <= x <= endi.

Example 1:

Input: ranges = [[1,2],[3,4],[5,6]], left = 2, right = 5
Output: true
Explanation: Every integer between 2 and 5 is covered:

- 2 is covered by the first range.
- 3 and 4 are covered by the second range.
- 5 is covered by the third range.
  Example 2:

Input: ranges = [[1,10],[10,20]], left = 21, right = 21
Output: false
Explanation: 21 is not covered by any range.

Constraints:

1 <= ranges.length <= 50
1 <= starti <= endi <= 50
1 <= left <= right <= 50

```js
var isCovered = function (ranges, left, right) {
  ranges.sort((a, b) => a[0] - b[0]);
  let res = [];
  let first = ranges[0];
  for (let i = 1; i < ranges.length; i++) {
    if (ranges[i][0] - 1 <= first[1]) {
      first[1] = Math.max(first[1], ranges[i][1]);
    } else {
      res.push(first);
      first = ranges[i];
    }
  }
  res.push(first);
  for (let range of res) {
    if (range[0] <= left && range[1] >= right) return true;
  }
  return false;
};
```
