-- “融合MMoE、ESMM等模型，对京东泰国站商品的点击、加购物车进行多目标预估，同时加入PCGrad进行梯度的裁剪和投影，来减少任务之间的冲突。 ”

**模型结构：**

将所有的特征都做embedding，然后一路做FM（输出一个logit），一路做DCN-v2（输出embedding），若干路做MLP（输出embedding）。之前还考虑过xDeepFM和AutoInt，但是由于复杂度太高（xDeepFM复杂度是立方级别，AutoInt复杂度是平方级别），并没有上线。

对于两个任务（是否点击/是否加购），用两个gate，分别以特征embedding作为输入，输出若干个attention权重，去给expert的输出结果做加权。然后，把这个不同的加权结果输入到ctr预估tower和cvr预估tower，分别得到两个logit，这就是一个item被点击的概率 & 点击之后加购物车的概率。那么，为了解决sampled bias问题，我们直接求点击and加购物车概率pctcvr，来作为cvr的一个辅助loss。

**采样方法：**

选择近期某7天在首页推荐模块的用户点击log，然后去hive表中取用户属性字段、item属性字段。user属性字段除了一些用户的人口属性特征（其中缺失字段进行补全），还采用了用户的行为特征：行为特征是选择了**统计**的方法，比如“该用户近7天内对**此item**对应三级品类的浏览次数”等等。此外，还有一些人工打的标签，这些都是根据一些统计的方法做的（毕竟我以前是BI工程师...），作为泰国站的BI工程师，我们就是要参考国内版本的业务逻辑，和国内的标签对标。最后生成兴趣标签（如“咖啡爱好者”等，我的标签还有“音乐爱好者”）、用户价值标签、用户价格敏感度标签等。

item的特征有价格、类别属性等特征，还有近50天点击数等交互特征。

ctr的正样本显然就是点击样本了，那么负样本我们要注意采“真负例”，即用户真正不感兴趣的内容，防止引入噪声。具体方法就是取用户最后一个点击的item之前那些没被点击的样本当作负例 -- 这些是用户真正不感兴趣的内容。

cvr（加购物车）的正例是在首次点击之后24h之内加购物车的样本。由于加购的延迟不如购买那么大，所以不需要考虑延迟转化的问题。

[![img](https://camo.githubusercontent.com/aa5954c7d33a79a29d27023aa4b098448fde7cc8acd80faf786848061cca275e/68747470733a2f2f706963322e7a68696d672e636f6d2f38302f76322d39333534656261653631633634376334636130353065333061656162643331355f31343430772e706e67)](https://camo.githubusercontent.com/aa5954c7d33a79a29d27023aa4b098448fde7cc8acd80faf786848061cca275e/68747470733a2f2f706963322e7a68696d672e636f6d2f38302f76322d39333534656261653631633634376334636130353065333061656162643331355f31343430772e706e67)

**PCGrad：**

caveat: 其实这篇文章是很有问题的，梯度的conflict不一定是个坏事，反而能够起到正则化的效果；而文章除了做方向的de-conflict之外，还做了模长的归一化，那么会不会是因为模长归一化而导致产生了类似梯度裁剪的效果呢？文章并没有进行相关的消融实验。

PCGrad是一个可插拔的优化器，它对普通的优化器（如Adam）又进行了一步封装：

```
import torch
import torch.nn as nn
import torch.optim as optim
from pcgrad import PCGrad

# wrap your favorite optimizer
optimizer = PCGrad(optim.Adam(net.parameters())) 
losses = [...] # a list of per-task losses
assert len(losses) == num_tasks
optimizer.pc_backward(losses) # calculate the gradient can apply gradient modification
optimizer.step()  # apply gradient step
class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction ##self._optim就是原本的优化器
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True) ##调用原始的优化器进行zero_grad()

    def step(self):
        return self._optim.step() ##调用原始优化器的step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives) ##获得每个task对应loss的梯度
        pc_grad = self._project_conflicting(grads, has_grads) ##去掉conflict的部分，同时做了梯度裁剪
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad) ##修改grad
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
```