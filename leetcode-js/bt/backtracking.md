### 1 78. Subsets

```js
var subsets = function(nums) {
    const res = [];
    const subsets = [];
    const dfs = (i) => {
        if (i === nums.length) {
            res.push([...subsets]);
            return;
        }
        subsets.push(nums[i]);
        dfs(i + 1);
        subsets.pop();
        dfs(i + 1);
    }
    dfs(0);
    return res;
};
```