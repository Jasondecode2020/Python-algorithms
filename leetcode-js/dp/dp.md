### 1 509. Fibonacci Number

method 1: recursion

```js
var fib = function(n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
};
```

method 2: dp

```js
var fib = function(n) {
    const dp = new Array(n + 1).fill(0);
    dp[1] = 1;
    for (let i = 2; i < n + 1; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
};
```

### 2 1137. N-th Tribonacci Number

```js
var tribonacci = function(n) {
    const dp = new Array(n + 1).fill(0);
    dp[1] = 1;
    dp[2] = 1;
    for (let i = 3; i < n + 1; i++) {
        dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3];
    }
    return dp[n];
};
```

### 313. Super Ugly Number

```js
var nthSuperUglyNumber = function(n, primes) {
    // start dp from index 1, multiples also start from index 1
    const multiples = new Array(primes.length).fill(1);
    const dp = new Array(n + 1).fill(1);
    for (let i = 2; i < n + 1; i++) {
        dp[i] = Infinity;
        let currIndex = [];
        for (let j = 0; j < primes.length; j++) {
            const minValue = dp[multiples[j]] * primes[j];
            if (minValue < dp[i] && minValue > dp[i - 1]) {
                dp[i] = minValue;
                currIndex = [j];
            } else if (minValue === dp[i]) {
                currIndex.push(j);
            }
        }
        // increase of all indice if have the same value
        for (let i = 0; i < currIndex.length; i++) {
            multiples[currIndex[i]]++;
        }
    }
    return dp[n];
};
```

300. Longest Increasing Subsequence

```js
var lengthOfLIS = function(nums) {
    const dp = new Array(nums.length).fill(1);
    for (let i = 1; i < nums.length; i++) {
        for (let j = i - 1; j >= 0; j--) {
            if (nums[i] > nums[j]) dp[i] = Math.max(dp[i], dp[j] + 1);
        }
    }
    return Math.max(...dp);
};
```