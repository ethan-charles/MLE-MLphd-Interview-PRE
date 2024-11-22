# Self-Attention 和 Cross-Attention 的区别

---

## 1. Self-Attention
- **定义**：在 **self-attention** 中，查询（Query）、键（Key）、和值（Value）都来自同一个序列。
- **应用场景**：
  - 用于同一序列中不同位置之间的关系建模。
  - 例如：Transformer 编码器的每一层都使用 self-attention，处理输入序列的特征并捕捉其内部依赖关系。
- **机制**：
  - 计算公式：  
    \[
    \text{Attention(Q, K, V)} = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
    \]
    其中 \( Q, K, V \) 都是从同一个输入序列中生成。
  - 模型通过计算不同位置之间的相似性（相关性分数），聚合相关的信息。
- **优势**：
  - 捕捉序列中的长距离依赖关系。
  - 提取同一输入序列中的上下文信息。

---

## 2. Cross-Attention
- **定义**：在 **cross-attention** 中，查询（Query）来自一个序列，而键（Key）和值（Value）来自另一个序列。
- **应用场景**：
  - 用于两个不同序列之间的关系建模。
  - 例如：Transformer 解码器使用 cross-attention，从编码器生成的表示（作为 Key 和 Value）中提取信息，结合解码器的输入（作为 Query）生成输出。
- **机制**：
  - 计算公式与 self-attention 相同：  
    \[
    \text{Attention(Q, K, V)} = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
    \]
    但这里 \( Q \) 来自解码器输入，而 \( K \) 和 \( V \) 来自编码器输出。
  - 模型通过跨序列的相似性计算，从另一个序列中提取最相关的信息。
- **优势**：
  - 融合两个序列的信息，适用于编码器-解码器架构中的序列到序列（seq2seq）任务。

---

## 核心区别

| 特性               | **Self-Attention**                                  | **Cross-Attention**                                  |
|--------------------|---------------------------------------------------|----------------------------------------------------|
| **Query, Key, Value来源** | 同一序列                                        | 不同序列                                           |
| **目标**            | 建模序列内部的依赖关系                             | 在两个序列之间提取相关信息                          |
| **应用场景**        | 编码器中常见，或者在解码器中自身序列的依赖建模       | 编解码器之间的信息交互                              |
| **典型任务**        | 文本上下文建模、图像特征提取等                      | 机器翻译、跨模态任务（如图像描述生成）等            |

---

## 例子
1. **Self-Attention**:
   - 输入为句子 “I love machine learning”，计算每个词与其他词的相关性以捕捉句法和语义信息。
2. **Cross-Attention**:
   - 在机器翻译中，输入法语句子 “J’aime l’apprentissage automatique”，解码器（生成英语翻译）会用 cross-attention 从编码器的法语表示中提取关键信息。

---

## 结合使用
- **Self-Attention** 捕捉单一序列的全局上下文。
- **Cross-Attention** 融合不同序列的信息。
