### 1 738. Monotone Increasing Digits

```js
var monotoneIncreasingDigits = function(n) {
    const arr = (n + '').split('');
    let j = arr.length; // find the index where fill in '9'
    for (let i = arr.length - 1; i > 0; i--) {
        if (arr[i - 1] <= arr[i]) continue;
        arr[i - 1]--;
        j = i;
    }
    for (let i = j; i < arr.length; i++) arr[i] = '9';
    return arr.join('');
};
```

### 2 2160. Minimum Sum of Four Digit Number After Splitting Digits

```js
var minimumSum = function(num) {
    n = num.toString().split('').map(Number).sort();
    return Number(`${n[0]}${n[2]}`) + Number(`${n[1]}${n[3]}`);
};
```

### 3 1974. Minimum Time to Type Word Using Special Typewriter

```js
var minTimeToType = function(word) {
    let res = 0;
    let curr = 'a';
    for (let c of word) {
        let tempDist = Math.abs(c.charCodeAt(0) - curr.charCodeAt(0));
        const dist = Math.min(tempDist + 1, 26 - tempDist + 1);
        res += dist;
        curr = c;
    }
    return res;
};
```

### 4 2027. Minimum Moves to Convert String

```js
var minimumMoves = function(s) {
    let i = 0;
    res = 0;
    while (i < s.length) {
        if (s[i] === 'X') {
            s[i] = 'O';
            s[i + 1] = 'O';
            s[i + 2] = 'O';
            i += 3;
            res += 1;
        } else {
            i++;
        }
    }
    return res;
};
```

### 5 2078. Two Furthest Houses With Different Colors

```js
var maxDistance = function(colors) {
    let res = 0;
    for (let i = 0; i < colors.length; i++) {
        for (let j = i + 1; j < colors.length; j++) {
            if (colors[j] !== colors[i] && j - i > res) res = j - i;
        }
    }
    return res;
};
```

two pointers
```js
var maxDistance = function(colors) {
    let l = 0, r = colors.length - 1;
    while(colors[r] === colors[l]) l++;
    while(colors[0] === colors[r]) r--;
    return Math.max(r, colors.length - 1 - l);
};
```