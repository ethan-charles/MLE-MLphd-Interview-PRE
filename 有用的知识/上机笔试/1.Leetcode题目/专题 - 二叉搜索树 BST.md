# 专题 - 二叉搜索树 (Binary Search Tree)

[TOC]



### 1. 利用中序遍历

二叉搜索树的一个重要性质：中序遍历是单调递增的。我们就可以利用中序遍历搞事情！比如在中序遍历的过程中记录一些变量。



#### [99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/)

难度中等699

给你二叉搜索树的根节点 `root` ，该树中的 **恰好** 两个节点的值被错误地交换。*请在不改变其结构的情况下，恢复这棵树* 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/10/28/recover2.jpg)

```
输入：root = [3,1,4,null,null,2]
输出：[2,1,4,null,null,3]
解释：2 不能在 3 的右子树中，因为 2 < 3 。交换 2 和 3 使二叉搜索树有效。
```

题解：

我们在中序遍历的时候，**顺便**记录下来哪一块位置的节点是**有问题的节点**，把两个节点都找出来之后交换二者的值即可。

题目给出的例子不是很有代表性，我们来自己写一个有代表性的例子：

![img](https://pic2.zhimg.com/80/v2-8eb4924c1b014f3fb9e8fcb5be4d3b27_1440w.png)

```python

class Solution:
    prev = None
    error1 = None ##第一个有问题的节点，为在前面的大数
    error2 = None ##第二个有问题的节点，在后面的小数
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def inorder(root): ##中序遍历
            if root == None: 
                return
            inorder(root.left) 
            ###开始处理当前的值###
            if not self.prev: 
                self.prev = root
            else:
                if self.prev.val > root.val and self.error1 == None: ##前面的大数，只记录第一个
                    self.error1 = self.prev
                if self.prev.val > root.val: ##后面的小数，不断更新
                    self.error2 = root
            self.prev = root
            ####################
            inorder(root.right) 
        inorder(root)
        #### 交换二者的值 ####
        tmp = self.error1.val
        self.error1.val = self.error2.val
        self.error2.val = tmp
```



#### [1038. 从二叉搜索树到更大和树](https://leetcode.cn/problems/binary-search-tree-to-greater-sum-tree/)

难度中等165

给定一个二叉搜索树 `root` (BST)，请将它的每个节点的值替换成树中大于或者等于该节点值的所有节点值之和。

提醒一下， *二叉搜索树* 满足下列约束条件：

- 节点的左子树仅包含键 **小于** 节点键的节点。
- 节点的右子树仅包含键 **大于** 节点键的节点。
- 左右子树也必须是二叉搜索树。

**示例 1：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/05/03/tree.png)**

```
输入：[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
```

题解：

我们只需对二叉搜索树进行**反向的中序遍历**，即右-根-左，然后记录后缀和，每次都修改节点的val。

对中序遍历的迭代法进行修改：

```python
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        stack = []  ##显式的用栈模拟
        sum_tot = 0
        tmp = root
        while(True): ##反复地
            while(tmp):  ##藤蔓入栈
                stack.append(tmp)
                tmp = tmp.right
            if len(stack) == 0: ##直至所有节点处理完毕
                break
            tmp = stack.pop()  ##x的右子树或为空，或已经遍历，故可以...
            sum_tot += tmp.val##...立即访问之
            tmp.val = sum_tot ##这里修改
            tmp = tmp.left ##再转向左子树
        return root
```



### 2. 判断一个树是否是二叉搜索树

二叉搜索树（BST）的特点：任一节点均不小于/不大于其左/右**后代**。

这并不等价于：任一节点都不小于/不大于其左/右**孩子**！！

#### [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)

难度中等1584

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

题解：

这道题的中序遍历方法自然好想，那么我们现在来讲一下递归方法。易错点！在递归的时候，我们不仅要判断根节点是不是 > 左节点，< 右节点，更重要的是要判断根节点是否 > 左节点中的最大值、< 右节点中的最小值。看下面这个例子：

![img](https://pic1.zhimg.com/80/v2-c5efa2cff76b886c4caa303085944627_1440w.png)

为了实现这个，我们**需要记录lower和upper**，如果root.val出现在[lower,upper]范围内，则可以；在递归左孩子的时候，我们把upper给设成root.val, 在递归右孩子的时候，我们把lower设成root.val. 

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helper(root,lower,upper):
            if not root:
                return True
            is_valid = root.val > lower and root.val < upper
            return is_valid and helper(root.left,lower,root.val) and helper(root.right,root.val,upper)
        return helper(root,-2**32,2**32)
```





#### 333. 最大BST子树

给定一个二叉树，找到其中最大的二叉搜索树（BST）子树，其中最大指的是子树节点数最多的。

**注意:**

子树必须包含其所有后代。

示例：

```text
示例:

输入: [10,5,15,1,8,null,7]

   10 
   / \ 
  5  15 
 / \   \ 
1   8   7

输出: 3
解释: 高亮部分为最大的 BST 子树。
     返回值 3 在这个样例中为子树大小。
```

进阶:

你能想出用 O(n) 的时间复杂度解决这个问题吗？

题解：

首先，我们能够想到`判断一个树是否为二叉搜索树`的方法，这个在上面就已经讲过了（重点是记录lower和upper）。那么，一个最简单的想法就是，遍历所有的节点，判断该节点是否是二叉搜索树，然后返回具有最大节点数的那个树。但是这样，我们需要遍历完n个节点，每个节点还都要判断是否是二叉搜索树，所以复杂度是O(n^2)的。

那么，能不能在判断一个节点是否是BST的同时，也记录每个节点的子树大小、并用一个全局变量来保存最大的BST子树节点个数呢？

这样，一个dfs函数就需要具有**两个返回值**。

```python
max_num = 0 ##最大BST子树的节点个数，这是一个全局变量
def Solution(root):  ##寻找最大的BST子树
    def dfs(root,lower,upper): ##判断是否是BST，同时记录子树大小
        global max_num
        if root == None:
            return True, 0
        is_bst = root.val > lower and root.val < upper
        left_is_bst, left_node_num = dfs(root.left,lower,root.val) ##左子树必须都<root.val
        right_is_bst, right_node_num = dfs(root.right,root.val, upper) ##右子树必须都> root.val

        if left_is_bst: ##左子树是BST
            max_num = max(max_num,left_node_num) ##更新
        if right_is_bst:##右子树是BST
            max_num = max(max_num,right_node_num)##更新
        is_bst = is_bst and left_is_bst and right_is_bst
        node_num = 1+left_node_num+right_node_num ##节点数
        if is_bst:##我是BST
            max_num = max(max_num,node_num)##更新
        return is_bst,node_num ##返回两个值：是否是BST、节点个数
    dfs(root,-2**32,2**32)
    return max_num
```



### 3. 二叉搜索树的查询/插入/删除

#### 3.1 查找

从根节点出发，逐步的缩小查找范围，直到：

- 发现目标（成功）
- 发现None（失败）

整个过程是在仿造有序向量的**二分查找**。

递归解法：

```python
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return None
        if root.val == val:
            return root
        elif root.val < val:
            return self.searchBST(root.right,val)
        else:
            return self.searchBST(root.left,val)
```

迭代解法：

```python
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if root.val == val:
                return root
            if root.val > val:
                root = root.left
            else:
                root = root.right
        return None
```



#### 3.2 插入

我们需要在BST中插入一个新的元素，而这个元素在原先的BST中不存在。这样，我们首先调用search，来找到**待插入位置hot**。之后，我们把新的元素插入到hot的左/右孩子即可。

![img](https://pic3.zhimg.com/80/v2-dd83cc791d3e9de1bbee09ec6b0fe250_1440w.png)

![img](https://pic2.zhimg.com/80/v2-4fa062b80b465f9c3c6828aec42cbeae_1440w.png)

```python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        hot = None
        if not root:
            return TreeNode(val)
        tmp = root
        while tmp:
            hot = tmp
            if tmp.val < val:
                tmp = tmp.right
            else:
                tmp = tmp.left
        if val < hot.val:
            hot.left = TreeNode(val)
        else:
            hot.right = TreeNode(val)
        return root
```



#### 3.3 删除

如果待删除节点只有一个孩子/或者没有孩子：直接删除这个节点即可，然后用其唯一的孩子代替该节点

![img](https://pic1.zhimg.com/80/v2-a05517f91eb4a94a8d2ea0517921e455_1440w.png)

如果待删除节点有两个孩子：那么先把待删除节点和其中序后继交换（交换数值！），其中序后继就是它的右孩子的最左边那个节点，所以中序后继必没有左孩子。那么，这个问题就规约到了上面“只有一个孩子”的情况！

![img](https://pic3.zhimg.com/80/v2-1dad488a906227aa2ae49818665771a0_1440w.png)

```python
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        if root.val == key: ##找到
            if not root.left: ##只有一个孩子
                return root.right
            if not root.right:##只有一个孩子
                return root.left
            if root.left and root.right:##有两个孩子
                succ = root.right
                while succ.left:
                    succ = succ.left ##找到其中序后继
                ##交换root和中序后继
                tmp = succ.val 
                succ.val = root.val
                root.val = tmp
                ##################
                root.right = self.deleteNode(root.right, key)
                return root
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
            return root
        else:
            root.left = self.deleteNode(root.left,key)
            return root
```

易错：交换两个节点**的数值**，还是用tmp方法做交换，而不要直接写`succ, root = root,succ`. 



### 4. 二叉搜索树的期望树高分析

![img](https://pic2.zhimg.com/80/v2-7d71224ff31a39c3f307bc38face00a8_1440w.png)

![img](https://pic1.zhimg.com/80/v2-1c6f16bc4566378c2a352f3924bba1d8_1440w.png)

![img](https://pic1.zhimg.com/80/v2-cb5075e4de7565e1d4b45ca26c69ebc4_1440w.png)



#### [96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/)

难度中等1763

给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)

```
输入：n = 3
输出：5
```

题解：

这就是上面所说“随机组成”的情况，像拼积木一样把关键码“拼起来”，看看总共有多少种拓扑结构。

那么，对于1,2,3,...n 的关键码，我们在中间找到一个i：1,2,...i,...n 。以i为根节点，左边构成左子树、右边构成右子树，两者的情况相乘，就得到了以i为根的二叉搜索树个数。从1~n遍历所有的i，把结果加起来就得到了n个节点构成的二叉搜索树个数。

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0]*(n+1)
        dp[1] = 1
        dp[0] = 1
        for i in range(2,n+1):
            for j in range(1,i+1):
                dp[i] += dp[j-1]*dp[i-j] ##左边和右边相乘，因为这是组合。
        return dp[-1]
```

