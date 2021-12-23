### 1
1979. Find Greatest Common Divisor of Array

Given an integer array nums, return the greatest common divisor of the smallest number and largest number in nums.

The greatest common divisor of two numbers is the largest positive integer that evenly divides both numbers.

Example 1:

Input: nums = [2,5,6,9,10]
Output: 2
Explanation:
The smallest number in nums is 2.
The largest number in nums is 10.
The greatest common divisor of 2 and 10 is 2.

```js
var findGCD = function(nums) {
    const commonDivisor = (a, b) => {
        if (a === 0) return b
        return commonDivisor(b % a, a)
    }
    let numsMin = Math.min(...nums)
    let numsMax = Math.max(...nums)
    const res = commonDivisor(numsMax, numsMin)
    return res
};
```

### 2
2057. Smallest Index With Equal Value

Given a 0-indexed integer array nums, return the smallest index i of nums such that i mod 10 == nums[i], or -1 if such index does not exist.

x mod y denotes the remainder when x is divided by y.

Example 1:

Input: nums = [0,1,2]
Output: 0
Explanation: 
i=0: 0 mod 10 = 0 == nums[0].
i=1: 1 mod 10 = 1 == nums[1].
i=2: 2 mod 10 = 2 == nums[2].
All indices have i mod 10 == nums[i], so we return the smallest index 0.

```js
var smallestEqual = function(nums) {
    for (let i = 0; i < nums.length; i++) {
        if (i % 10 === nums[i]) return i
    }
    return -1
};
```

### 3 2053. Kth Distinct String in an Array

A distinct string is a string that is present only once in an array.

Given an array of strings arr, and an integer k, return the kth distinct string present in arr. If there are fewer than k distinct strings, return an empty string "".

Note that the strings are considered in the order in which they appear in the array.

 

Example 1:

Input: arr = ["d","b","c","b","c","a"], k = 2
Output: "a"
Explanation:
The only distinct strings in arr are "d" and "a".
"d" appears 1st, so it is the 1st distinct string.
"a" appears 2nd, so it is the 2nd distinct string.
Since k == 2, "a" is returned. 

```js
var kthDistinct = function(arr, k) {
    const map = new Map()
    for (let i = 0; i < arr.length; i++) {
        if (!map.has(arr[i])) {
            map.set(arr[i], 1)
        } else {
            map.set(arr[i], map.get(arr[i]) + 1)
        }
    }
    
    let count = 0
    for (let [key, value] of map) {
        if (value === 1) count++
        if (count === k) return key
    }
    return ""
};
```

### 4 1403. Minimum Subsequence in Non-Increasing Order

Given the array nums, obtain a subsequence of the array whose sum of elements is strictly greater than the sum of the non included elements in such subsequence. 

If there are multiple solutions, return the subsequence with minimum size and if there still exist multiple solutions, return the subsequence with the maximum total sum of all its elements. A subsequence of an array can be obtained by erasing some (possibly zero) elements from the array. 

Note that the solution with the given constraints is guaranteed to be unique. Also return the answer sorted in non-increasing order.

Example 1:

Input: nums = [4,3,10,9,8]
Output: [10,9] 
Explanation: The subsequences [10,9] and [10,8] are minimal such that the sum of their elements is strictly greater than the sum of elements not included, however, the subsequence [10,9] has the maximum total sum of its elements. 

```js
var minSubsequence = function(nums) {
    const sum = nums.reduce((acc, curr) => acc + curr, 0)
	let subSum = 0
	nums.sort((a, b) => b - a)

	for (const index in nums) {
		subSum += nums[index]
		if (subSum > sum - subSum) return nums.slice(0, +index + 1) // convert to number
	}
};
```

### 5 2032. Two Out of Three

Given three integer arrays nums1, nums2, and nums3, return a distinct array containing all the values that are present in at least two out of the three arrays. You may return the values in any order.

Example 1:

Input: nums1 = [1,1,3,2], nums2 = [2,3], nums3 = [3]
Output: [3,2]
Explanation: The values that are present in at least two arrays are:
- 3, in all three arrays.
- 2, in nums1 and nums2.

```js
var twoOutOfThree = function(nums1, nums2, nums3) {
    const res = []
    const set1 = new Set(nums1)
    const set2 = new Set(nums2)
    const set3 = new Set(nums3)
	
    for (let s of set1){
        if (set2.has(s)){
           res.push(s)
        }
        if (set3.has(s)){
           res.push(s)
        }
    }
    
    for (let s of set2){
        if (set3.has(s)){
           res.push(s)
        }
    }
    
    return [...new Set(res)]
};
```