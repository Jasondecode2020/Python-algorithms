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

### 2 1239. Maximum Length of a Concatenated String with Unique Characters

```python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        charSet = set()
        
        def overlap(charset, s):
            c = Counter(charSet) + Counter(s)
            return max(c.values()) > 1
        
        def backtrack(i):
            if (i == len(arr)):
                return len(charSet)
            
            res = 0
            if not overlap(charSet, arr[i]):
                for c in arr[i]:
                    charSet.add(c)
                res = backtrack(i + 1)
                for c in arr[i]:
                    charSet.remove(c)
            return max(res, backtrack(i + 1))
        return backtrack(0)
```