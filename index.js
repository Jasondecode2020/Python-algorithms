const dp = [1, 2, 3]
const f = (dp) => {
    for (let i = 0; i < dp.length; i++) {
        if (i === 0) {
            dp[i] = 2;
            return dp[i];
        }
    }
}


const a = f(dp);
console.log(a);
