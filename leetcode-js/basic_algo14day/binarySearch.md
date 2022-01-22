### 1 704. Binary Search

```js
var search = function(nums, target) {
    let l = 0;
    let r = nums.length - 1;
    while (l <= r) {
        const m = Math.floor((l + r) / 2);
        if (nums[m] > target) r = m - 1;
        else if (nums[m] < target) l = m + 1;
        else return m;
    }
    return -1;
};
```

### 2 278. First Bad Version

```js
var solution = function(isBadVersion) {
    /**
     * @param {integer} n Total versions
     * @return {integer} The first bad version
     */
    return function(n) {
        let l = 0;
        let r = n;
        while (l <= r) {
            const m = Math.floor((l + r) / 2);
            if (!isBadVersion(m)) l = m + 1;
            else r = m - 1;
        }
        return l;
    };
};
```

### 3 35. Search Insert Position

```js
var searchInsert = function(nums, target) {
    let l = 0;
    let r = nums.length - 1;
    while (l <= r) {
        const m = Math.floor((l + r) / 2);
        if (nums[m] > target) r = m - 1;
        else if (nums[m] < target) l = m + 1;
        else return m;
    }
    return l;
};
```

### 4 2089. Find Target Indices After Sorting Array

```js
// O(nlog(n))
var targetIndices = function(nums, target) {
    const res = [];
    nums.sort((a, b) => a - b);
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] === target) res.push(i);
        if (nums[i] > target) break;
    }
    return res;
};
```

```js
// O(nlog(n))
// find numbers lower than target and find total matching numbers
// then just put them in array
var targetIndices = function(nums, target) {
    let lowerCount = 0;
    let matchingCount = 0;
    
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] < target) {
            lowerCount++;
        } else if (nums[i] === target) {
            matchingCount++;
        }
    }
    
    let result = [];
    for (let i = lowerCount; i < lowerCount + matchingCount; i++) {
        result.push(i);
    }
    
    return result;
};
```

### 5 1539. Kth Missing Positive Number

```js
var findKthPositive = function(arr, k) {
    let count = 0;
    const s = new Set(arr);
    for (let i = 1; i < Infinity; i++) {
        if (!s.has(i)) count++;
        if (count === k) return i;
    }
};
```