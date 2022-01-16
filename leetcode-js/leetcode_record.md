### 1 1. Two Sum

code: 

```js
var twoSum = function(nums, target) {
    const map = new Map();
    for (let i = 0; i < nums.length; i++) {
        const res = target - nums[i];
        if (map.has(res)) return [map.get(res), i];
        map.set(nums[i], i);
    }
};
```

time (minute): 2

note: did many times, use hashmap to store value : index first
then find the res in hashmap

### 2 2. Add Two Numbers

code: 

```js
var addTwoNumbers = function(l1, l2) {
    const dummy = new ListNode();
    let p = dummy;
    let carry = 0;
    while (l1 || l2) {
        const val = (l1 ? l1.val : 0) + (l2 ? l2.val : 0) + carry;
        p.next = new ListNode(val % 10);
        p = p.next;
        if (l1) l1 = l1.next;
        if (l2) l2 = l2.next;
        carry = parseInt(val / 10);
    }
    if (carry) p.next = new ListNode(1);
    return dummy.next;
};
```

time (minute): 4

note: did many times, basic use of linked list with a carry

### 3 3. Longest Substring Without Repeating Characters

code: 

```js
var lengthOfLongestSubstring = function(s) {
    const map = new Map();
    let left = 0;
    let longest = 0;
    for (let right = 0; right < s.length; right++) {
        if (map.has(s[right])) left = Math.max(left, map.get(s[right]) + 1);
        map.set(s[right], right);
        longest = Math.max(longest, right - left + 1);
    }
    return longest;
};
```

time (minute): 7

note: did many times, but still made a mistake, by debugging, used long time
basic use of sliding window as a set.keep characters in sliding window always be
unique.

### 5 5. Longest Palindromic Substring

code: 

```js
var longestPalindrome = function(s) {
    const helper = (l, r) => {
        while (l >= 0 && r < s.length && s[l] === s[r]) {
            l--;
            r++;
        }
        return s.slice(l + 1, r);
    }
    
    let res = '';
    for (let i = 0; i <s.length; i++) {
        let test = helper(i, i);
        if (test.length > res.length) res = test;
        test = helper(i, i + 1);
        if (test.length > res.length) res = test;
    }
    return res;
};
```

time (minute): 5

note: did many times, this is basically a brute force method
but by using helper function(expand from center) start from 
middle makes it fast, but still O(n^2), there is a Manacher's
algorithm but too hard for an interview.

### 6 6. Zigzag Conversion

code: 

```js
var convert = function(s, numRows) {
    if (numRows === 1 ||numRows >= s.length) return s; // no need to use bucket
    // use bucket to get result
    let bucket = new Array(numRows).fill().map(i => []);
    let flip = -1; // as a plus minus flip
    let count = 0; // count which bucket to put
    for (let i = 0; i < s.length; i++) {
        if (count === 0) flip = -flip;
        bucket[count].push(s[i]);
        count += flip;
        if (count === numRows - 1) flip = -flip;
    }
    
    for (let i = 0; i < bucket.length; i++) {
        bucket[i] = bucket[i].join('');
    }
    return bucket.join('')
};
```

time (minute): 5

note: did many times, use bucket to store each row
### 7 7. Reverse Integer

code: 

```js
var reverse = function(x) {
    const reversePositiveInteger = (n) => {
        let ans = 0;
        while (n) {
            let res = n % 10;
            ans = res + ans * 10;
            n = parseInt(n / 10);
        }
        return ans;
    }
    
    let res = 0;
    if (x >= 0) res = reversePositiveInteger(x);
    else res = -reversePositiveInteger(-x);
    
    if (res >= -Math.pow(2, 31) && res <= Math.pow(2, 31) - 1) return res;
    return 0;
};
```

time (minute): 5

note: did many times, normal reverse number.

### 8 8. String to Integer (atoi)

code: 

```js
var myAtoi = function(s) {
    let i = 0,
        num = 0,
        max = 2 ** 31 - 1,
        min = 2 ** 31 * -1
        sign = 1;
    s = s.trim();
    if (s[i] === '+' || s[i] === '-') {
        sign = s[i] === '-' ? -1 : 1;
        i++;
    }
    while (s[i] && s[i].charCodeAt(0) - 48 >= 0 && s[i].charCodeAt(0) - 48 <= 9) {
        num = num * 10 + s[i].charCodeAt(0) - 48;
        i++;
    }
    num *= sign;
    return Math.max(Math.min(num, max), min)
};
```

time (minute): 5

note: did many times, did yesterday and today again.

### 9 9. Palindrome Number

code: 

```js
var isPalindrome = function(x) {
    const reversePositiveInteger = (n) => {
        let ans = 0;
        while (n) {
            let res = n % 10;
            ans = res + ans * 10;
            n = parseInt(n / 10);
        }
        return ans;
    }
    
    if (x < 0) return false;
    return reversePositiveInteger(x) === x;
};
```

time (minute): 5

note: did many times, used leetcode 7 method

### 11 11. Container With Most Water

code: 

```js
var maxArea = function(height) {
    let l = 0;
    let r = height.length - 1;
    let res = 0;
    while (l < r) {
        res = Math.max(res, (r - l) * Math.min(height[r], height[l]));
        if (height[l] < height[r]) l++;
        else r--;
    }
    return res;
};
```

time (minute): 5

note: did many times, used two pointers

### 12 12. Integer to Roman

code: 

```js
var intToRoman = function(num) {
    const values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
    const romans = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"];
    let res = "";
    for (let i = 0; i < values.length; i++) {
        let count = parseInt(num / values[i]);
        while (count--) res += romans[i];
        num %= values[i];
    }  
    return res;
};
```

time (minute): 5

note: did many times, similar to leetcode 7 method

### 13 13. Roman to Integer

code: 

```js
var romanToInt = function(s) {
    const map = {
        'I': 1,           
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }
    let res = 0;
    let prev = map['M'];
    for (const c of s) {
        res += map[c];
        if (map[c] > prev) res -= 2 * prev;
        prev = map[c];
    }
    return res
};
```

time (minute): 5

note: did many times, used two pointers

### 14 14. Longest Common Prefix

code: 

```js
var longestCommonPrefix = function(strs) {
    let prefix = '';
    for (let i = 1; i <= strs[0].length; i++) {
        const nextPrefix = strs[0].slice(0, i);
        if (strs.every(str => str.slice(0, i) === nextPrefix)) {
            prefix = nextPrefix;
        } else {
            break;
        }
    }
    
    return prefix;
};
```

time (minute): 5

note: did many times, every is better

### 15 15. 3Sum

code: 

```js
var threeSum = function(nums) {
    nums.sort((a, b) => a - b);
    const res = [];
    for (let i = 0; i < nums.length; i++) {
        if (i > 0 && nums[i] === nums[i - 1]) continue;
        if (nums[i] > 0) break;
        let l = i + 1;
        let r = nums.length - 1
        while (l < r) {
            const sum = nums[i] + nums[l] + nums[r];
            if (sum === 0) {
                res.push([nums[i], nums[l], nums[r]])
                l++;
                r--;
                while (l < r) {
                    if (nums[l] === nums[l - 1]) l++;
                    else if (nums[r] === nums[r + 1]) r--;
                    else break;
                }
            } else if (sum > 0) {
                r--;
            } else {
                l++;
            }
        }
    }
    return res;
};
```

time (minute): 5

note: did many times, two pointers

### 849. Maximize Distance to Closest Person

```js
var maxDistToClosest = function(seats) {
    const closest = (i, j) => {
        while (i >= 0 && j < seats.length) {
            if (seats[i] === 0) i--;
            if (seats[j] === 0) j++;
            if (seats[i] === 1 && seats[j] === 1) return (j - i) >> 1; // 1 000 1
        }
        return j - i - 1; // 10000 || 00001
    }
    
    let res = 0;
    for (let i = 0; i < seats.length; i++) {
        if (seats[i] === 0) {
            const test = closest(i, i);
            if (test > res) res = test;
        }
    }
    return res;
};
```
time (minute): 5

note: did many times, two pointers