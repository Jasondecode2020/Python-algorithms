### 1 844. Backspace String Compare

```js
var backspaceCompare = function(s, t) {
    const handleString = (str) => {
        const stack = [];
        for (let c of str) {
            if (c !== '#') stack.push(c);
            else stack.pop();
        }
        return stack.join('');
    }
    return handleString(s) === handleString(t);
};
```