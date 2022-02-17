### 1 451. Sort Characters By Frequency

```js
var frequencySort = function(s) {
    const map = new Map();
    for (let c of s) map.has(c) ? map.set(c, map.get(c) + 1) : map.set(c, 1);
    const sortedString = new Map([...map.entries()].sort((a, b) => b[1] - a[1]));
    let res = "";
    for(const item of sortedString.keys()){
        res += item.repeat(map.get(item))
    }
    return res;
};
```