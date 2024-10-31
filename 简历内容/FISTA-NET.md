# 介绍FISTA-Net

FISTA-Net is composed of several key modules that work together for enhanced image reconstruction:

1. **Gradient Descent Module**: This layer performs gradient descent on the data fidelity term, enforcing consistency with observed data.
2. **CNN-based Proximal Mapping**: Acting as a learned proximal operator, this module improves image quality by applying learned regularization, replacing traditional soft-thresholding.
3. **Iterative Update Mechanism**: This component iterates updates using FISTA’s momentum, speeding convergence by balancing fidelity and regularization.

Each iteration improves the reconstruction by combining these modules, resulting in high-quality outputs suitable for inverse problems in imaging.

---

## FISTA和Iterative Shrinkage/Thresholding Algorithm (ISTA)的区别

**FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)** improves upon **ISTA (Iterative Shrinkage-Thresholding Algorithm)** by adding a momentum term that accelerates convergence. While ISTA only applies simple iterative updates with soft-thresholding, FISTA introduces a "momentum" or extrapolation term, which helps the solution reach convergence faster, especially for large-scale or high-dimensional problems. This acceleration makes FISTA especially advantageous when dealing with complex inverse problems in imaging or other computationally intense applications.

---

## Momentum 在优化算法中的作用

"Momentum" 在优化算法中是一种技术，用于加速收敛速度。它通过结合前一步更新的方向和幅度，来影响当前的更新，平滑迭代过程，减少震荡现象。在 **FISTA** 中，momentum 用于“外推”每一步的更新，从而比普通的 ISTA 更快收敛，这在高维或复杂问题中尤其有效。

---

## CNN的作用

In **FISTA-Net**, the convolutional neural network (**CNN**) component is used as a learned proximal operator. It plays a critical role in refining image quality by effectively mapping noisy or incomplete reconstructions to cleaner images. This CNN-based proximal mapping helps in regularizing and enhancing the reconstruction at each iteration, which leads to better visual fidelity and accuracy in the final output, especially in challenging imaging scenarios such as low-resolution or noisy data.

---

## Soft Thresholding 的作用

**Soft thresholding** is effective in image processing, particularly for denoising and sparse representation, due to the following reasons:

1. **Smooth Noise Reduction**: Soft thresholding reduces small noise values towards zero without completely removing them, resulting in smoother, more natural images than hard thresholding.
2. **Sparse Representation**: It preserves large coefficients (significant image features) while compressing smaller ones (noise), making it ideal for retaining important image details while suppressing noise.
3. **Balance Between Detail and Noise Suppression**: Adjustable thresholds allow soft thresholding to balance noise reduction and detail preservation, crucial for natural images.
4. **Multi-Scale Adaptability**: By applying soft thresholding across different scales (e.g., wavelet decomposition), it effectively denoises both fine details and larger structures.

---

## FISTA-Net 对 FISTA 的提升

**FISTA-Net** builds on **FISTA** by integrating a CNN-based learned proximal mapping rather than using a fixed thresholding operator. This modification allows FISTA-Net to adapt more flexibly to various data characteristics, effectively regularizing complex patterns and noise structures within images. The CNN enhances the model’s ability to recover high-quality reconstructions across different noise levels and resolutions, improving upon FISTA's speed and generalization while retaining its accelerated convergence through the momentum term.

---

## FISTA-Net 主要的公式

The proposed **FISTA-Net** to solve is formulated as:

1. \( r^{(k)} = y^{(k)} - W^{(k)^T} A y^{(k)} - b \)  
2. \( x^{(k)} = T^{(k)} (r^{(k)}) \)  
3. \( y^{(k+1)} = x^{(k)} + \eta^{(k)} (x^{(k)} - x^{(k-1)}) \)

---

## 空间分辨率的测量：边缘锐利度

**边缘锐利度**是指图像边缘的清晰度。高分辨率图像的边缘应该是锐利的，而不是模糊或“拖尾”的。边缘锐利度可以反映重建算法的分辨能力，因为它揭示了算法在重建过程中是否丢失了原始数据中的边界信息。

一种常用的边缘锐利度测量方法是计算边缘梯度。可以通过梯度算子（如 **Sobel 算子**）来计算边缘的梯度强度，梯度越高，边缘越锐利，从而表明空间分辨率越高。此外，边缘轮廓的宽度（例如通过 **半最大高宽度，Full Width at Half Maximum (FWHM)**）也可以用来判断边缘锐利度——边缘越窄，表示分辨率越高。

---

## ReLU 在 FISTA 中的作用

**ReLU** 是一种简单而有效的非线性激活函数，能够快速增加网络的非线性表达能力，同时避免了梯度消失问题。这样可以在 **FISTA（快速迭代收缩阈值算法）** 的迭代过程中，使模型更快地收敛到理想的解。

---

## FISTA-NET 的核心任务：多视图数据的空间特征提取

**FISTA-NET** 的核心任务是从多视图数据中提取空间特征。**Conv3D（三维卷积）** 适合用来处理这种多视图三维数据，因为它可以在三维空间中提取特征，能够捕捉图像在三个维度（例如深度、高度和宽度）上的局部结构。

---

## 不建议加入 Pooling 操作的原因

1. **可能导致空间分辨率损失**：通过减少特征图的空间维度，可以降低计算量，但也可能导致图像细节信息的丢失。
2. **高分辨率需求**：FISTA-NET 的目标是精细地重建图像，并保留高分辨率的细节。引入 **Pooling** 可能导致细小特征（如边缘和纹理）被模糊化，降低了重建质量。
