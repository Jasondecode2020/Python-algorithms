### 797. All Paths From Source to Target

```js
var allPathsSourceTarget = function(graph) {
    const end = graph.length - 1;
    const dfs = (node, path, res) => {
        if (node === end) res.push(path);
        for (const nxt of graph[node]) {
            dfs(nxt, [...path, nxt], res);
        }
        return res;
    }
    
    return dfs(0, [0], []);
};
```

### 2 429. N-ary Tree Level Order Traversal

```js
var levelOrder = function(root) {
    if(!root) return [];
    let queue = [root];
    let res=[];
    while(queue.length > 0){
        let currentlevel = [];
        let len = queue.length;
        for(let i=0; i < len; i++){
            let tmp = queue.shift();
            currentlevel.push(tmp.val);
            queue.push(...tmp.children);
        }
        res.push(currentlevel);
    }
    return res;
};
```