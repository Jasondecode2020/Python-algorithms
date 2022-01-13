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