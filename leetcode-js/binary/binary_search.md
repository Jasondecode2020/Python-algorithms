### 1 875. Koko Eating Bananas

```js
var minEatingSpeed = function(piles, h) {
    let l = 1;
    let r = Math.max(...piles);
    let res = r;
    
    while (l <= r) {
        m = Math.floor((l + r) / 2);
        let hours = 0;
        for (let p of piles) {
            hours += Math.ceil(p / m);
        }
        if (hours <= h) {
            res = Math.min(res, m);
            r = m - 1;
        } else {
            l = m + 1;
        }
    }
    return res;
};
```