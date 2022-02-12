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
        let l = 1;
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

### 4 977. Squares of a Sorted Array

```js
var sortedSquares = function(nums) {
    const res = [];
    let l = 0;
    let r = nums.length - 1;
    while (l <= r) {
        const left = Math.pow(nums[l], 2);
        const right = Math.pow(nums[r], 2);
        if (left > right) {
            res.unshift(left);
            l++;
        } else {
            res.unshift(right);
            r--;
        }
    }
    return res;
};
```

### 5 189. Rotate Array

```js
var rotate = function(nums, k) {
    nums.reverse();
    k = k % nums.length;
    const reverse = (l, r) => {
        while (l < r) {
            [nums[l], nums[r]] = [nums[r], nums[l]];
            l++;
            r--;
        }
    }
    reverse(0, k - 1);
    reverse(k, nums.length - 1)
};
```

### 5 189. Rotate Array

```js
var rotate = function(nums, k) {
    nums.reverse();
    k = k % nums.length;
    const reverse = (l, r) => {
        while (l < r) {
            [nums[l], nums[r]] = [nums[r], nums[l]];
            l++;
            r--;
        }
    }
    reverse(0, k - 1);
    reverse(k, nums.length - 1)
};
```

### 6 283. Move Zeroes

```js
var moveZeroes = function(nums) {
    let l = 0;
    for (let r = 1; r < nums.length; r++) {
        if (nums[l] === 0 && nums[r] !== 0) {
            [nums[l], nums[r]] = [nums[r], nums[l]];
            l++;
        } else if (nums[l] === 0 && nums[r] === 0) {
            continue;
        } else if (nums[l] !== 0) l++;
    }
};
```

### 7 167. Two Sum II - Input Array Is Sorted

```js
var twoSum = function(numbers, target) {
    let l = 0;
    let r = numbers.length - 1;
    while (l < r) {
        if (numbers[l] + numbers[r] === target) return [l + 1, r + 1];
        else if (numbers[l] + numbers[r] > target) r--;
        else l++;
    }
};
```

### 8 167. Two Sum II - Input Array Is Sorted

```js
var twoSum = function(numbers, target) {
    let l = 0;
    let r = numbers.length - 1;
    while (l < r) {
        if (numbers[l] + numbers[r] === target) return [l + 1, r + 1];
        else if (numbers[l] + numbers[r] > target) r--;
        else l++;
    }
};
```

### 9 167. Two Sum II - Input Array Is Sorted

```js
var twoSum = function(numbers, target) {
    let l = 0;
    let r = numbers.length - 1;
    while (l < r) {
        if (numbers[l] + numbers[r] === target) return [l + 1, r + 1];
        else if (numbers[l] + numbers[r] > target) r--;
        else l++;
    }
};
```

### 10 167. Two Sum II - Input Array Is Sorted

```js
var twoSum = function(numbers, target) {
    let l = 0;
    let r = numbers.length - 1;
    while (l < r) {
        if (numbers[l] + numbers[r] === target) return [l + 1, r + 1];
        else if (numbers[l] + numbers[r] > target) r--;
        else l++;
    }
};
```

### 11 167. Two Sum II - Input Array Is Sorted

```js
var twoSum = function(numbers, target) {
    let l = 0;
    let r = numbers.length - 1;
    while (l < r) {
        if (numbers[l] + numbers[r] === target) return [l + 1, r + 1];
        else if (numbers[l] + numbers[r] > target) r--;
        else l++;
    }
};
```