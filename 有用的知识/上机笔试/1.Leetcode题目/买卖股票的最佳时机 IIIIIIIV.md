# 买卖股票的最佳时机 I/II/III/IV

#### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/) I (easy)

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

**示例 1：**

```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

题解：

只能购买并卖出**一次**。从左到右维护一个最小值即可！很简单吧。

#### 122. 买卖股票的最佳时机 II (medium)

给定一个数组 prices ，其中 prices[i] 表示股票第 i 天的价格。

在每一天，你可能会决定购买和/或出售股票。你在任何时候 **最多只能持有一股股票**。你也可以购买它，然后在同一天出售。
返回你能获得的最大利润 。

```python
输入: prices = [7,1,5,3,6,4]
输出: 7
```

**解法：**

可以购买和卖出**任意多次**。这是和I的差别。

因为题中说“最多只能持有一股股票”，所以不妨以每天持有股票的个数来分类。`dp[i][0]`表示第i天，不持有股票的收益；`dp[i][1]`表示第i天，持有股票的收益。想要持有股票，就必须花钱来买它，所以收益是要减去当前股票的价格的。在卖掉的时候，**收益要加上当前股票的价格**，因为卖出了就可以得到收益。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp_one = [0] * len(prices) ##当前持有一支股票
        dp_zero = [0] * len(prices) ##当前不持有股票
        dp_one[0] = -prices[0]
        for i in range(1,len(prices)):
            dp_one[i] = max(dp_one[i-1],dp_zero[i-1] - prices[i]) ##现在买入
            dp_zero[i] = max(dp_zero[i-1],dp_one[i-1] + prices[i]) ##现在卖出
        return max(max(dp_one),max(dp_zero))
```

我们看到，这里已经引入**自动机**的思想了，状态0和状态1如何相互转换。



### 714. 买卖股票的最佳时机含手续费

给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，**每笔交易你只需要为支付一次手续费**。

解法：

还是和上一题一样，有两个状态：持有1个股票、不持有股票。区别就是，在卖出的时候需要花掉手续费。我们规定，在**买入的时候**需要交手续费就完事儿了。

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        dp_one = [0] * len(prices) ##持有一支
        dp_zero = [0] * len(prices) ##不持有
        dp_one[0] = -prices[0]
        for i in range(1,len(prices)):
            dp_zero[i] = max(dp_zero[i-1],dp_one[i-1] + prices[i] - fee)
            dp_one[i] = max(dp_one[i-1],dp_zero[i-1] - prices[i])
        return dp_zero[-1] 
```



### 309. 最佳买卖股票时机含冷冻期

给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

**卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。**

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

解法：

其实，这是一个**有限自动机**。总共无非有三个状态：

- 状态0：持有0股，非冷冻期
- 状态1：持有0股，且在冷冻期 (刚刚卖掉)
- 状态2：持有1股，（必非冷冻期，因为是买入的状态）



```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0]*len(prices) for _ in range(3)]
        dp[0][0] = 0
        dp[2][0] = -prices[0]
        ans = 0
        for i in range(1,len(prices)):
            dp[0][i] = max(dp[0][i-1],dp[1][i-1])
            dp[1][i] = max(dp[1][i-1],dp[2][i-1]+prices[i])
            dp[2][i] = max(dp[2][i-1],dp[0][i-1]-prices[i])
            ans = max(ans,dp[0][i],dp[1][i],dp[2][i])
        return ans
```



### 123. 买卖股票的最佳时机 III [hard]

给定一个数组，它的第 `i` 个元素是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 **两笔** 交易。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**解法：**

题中的一个重要限制：“**最多可以完成两笔交易**”。受到上面题目的启发，想到用有限自动机来表示不同的状态，并画出状态转移图：

![img](https://pic1.zhimg.com/80/v2-3abd42ba6c29475ddd9ef060500b72ce_1440w.png)



一个非常非常容易错的地方是，状态2，4的初始化`dp[2][i],dp[4][i]`应该被初始化为`-price[0]`, 表示**现在买入了第0支股票**，之前还做过一次买入、卖出。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0] * len(prices) for _ in range(6)]
        dp[1][0] = dp[3][0] = dp[5][0] = -prices[0] ##【易错】这个初始状态容易忽略！
        ans = 0
        for i in range(1,len(prices)):
            dp[1][i] = max(dp[1][i-1],dp[0][i-1] - prices[i])
            dp[2][i] = max(dp[2][i-1],dp[1][i-1] + prices[i])
            dp[3][i] = max(dp[3][i-1],dp[2][i-1] - prices[i])
            dp[4][i] = max(dp[4][i-1],dp[3][i-1] + prices[i])
            dp[5][i] = max(dp[5][i-1],dp[4][i-1] - prices[i])
            ans = max(ans,dp[1][i],dp[2][i],dp[3][i],dp[4][i],dp[5][i])
        return ans
```



### 188. 买卖股票的最佳时机 IV [hard]

你最多可以完成 **k** 笔交易。

把上面的代码从k = 2改为支持任意k值即可。

