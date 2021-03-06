### 1 877. Stone Game

```python
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        dp = {}
        def dfs(l, r):
            if l > r:
                return 0
            if (l, r) in dp:
                return dp[(l, r)]
            even = True if (r - l) % 2 else False
            left = piles[l] if even else 0
            right = piles[r] if even else 0
            
            dp[(l, r)] = max(dfs((l + 1), r) + left, dfs(l, (r - 1)) + right)
            return dp[(l, r)]
        return dfs(0, len(piles) - 1) > (sum(piles)) // 2
```

[
    [True,  False,  False, False, False, False], 
    [True,  False,  False, False, False, False], 
    [True,  True,   True,  True,  True,  False], 
    [False, True,   True,  False, True,  False], 
    [False, False,  True,  True,  True,  True], 
    [False, False,  False, False, False, True]
]
