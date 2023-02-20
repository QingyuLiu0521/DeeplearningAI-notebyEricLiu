## 第三周：浅层神经网络(Shallow neural networks)
1. **符号定义** ：
   - $a^{[n]}$：第n层的激活值
   - $a^{[n]}_{i}$：第n层的第i个结点
2. 神经网络的表示
   1. **输入层**：包含了神经网络的输入；
   2. **隐藏层**：在训练集中，无法得知中间结点准确值的层级
   3. **输出层**：负责产生预测值
   
   例子：

   ![](images/L1_week3_5.png)
   > 一个两层神经网络
   $$
   a^{[1]} =
      \left[
         \begin{array}{ccc}
         a^{[1]}_{1}\\
         a^{[1]}_{2}\\
         a^{[1]}_{3}\\
         a^{[1]}_{4}\\
         \end{array}
         \right]
   $$
   - 隐藏层参数：($W^{[1]}$,$b^{[1]}$)，$W$为4x3的矩阵，而$b$是一个4x1的向量
   - 输出层参数：($W^{[2]}$,$b^{[2]}$)，$W$为1x4的向量，而$b$是一个1x1的向量 (1x4是因为隐藏层有四个隐藏层单元而输出层只有一个单元)
   > w参数矩阵行数：这一层节点个数
   >
   > w参数矩阵列数：上一层节点个数

3. 计算一个神经网络的输出
   
   隐藏层公式：
   $$
   \left[
         \begin{array}{c}
         z^{[1]}_{1}\\
         z^{[1]}_{2}\\
         z^{[1]}_{3}\\
         z^{[1]}_{4}\\
         \end{array}
         \right]
         =
      \overbrace{
      \left[
         \begin{array}{c}
         ...W^{[1]T}_{1}...\\
         ...W^{[1]T}_{2}...\\
         ...W^{[1]T}_{3}...\\
         ...W^{[1]T}_{4}...
         \end{array}
         \right]
         }^{W^{[1]}}
         *
      \overbrace{
      \left[
         \begin{array}{c}
         x_1\\
         x_2\\
         x_3\\
         \end{array}
         \right]
         }^{input}
         +
      \overbrace{
      \left[
         \begin{array}{c}
         b^{[1]}_1\\
         b^{[1]}_2\\
         b^{[1]}_3\\
         b^{[1]}_4\\
         \end{array}
         \right]
         }^{b^{[1]}}
   $$
   $$
   a^{[1]} =
      \left[
         \begin{array}{c}
         a^{[1]}_{1}\\
         a^{[1]}_{2}\\
         a^{[1]}_{3}\\
         a^{[1]}_{4}
         \end{array}
         \right]
         = \sigma(z^{[1]})
   $$
   ![w600](images/L1_week3_7.png)
4. 多样本向量化
   $$
   x =
      \left[
         \begin{array}{c}
         \vdots & \vdots & \vdots & \vdots\\
         x^{(1)} & x^{(2)} & \cdots & x^{(m)}\\
         \vdots & \vdots & \vdots & \vdots\\
         \end{array}
         \right]
   $$
   $$
   Z^{[1]} =
      \left[
         \begin{array}{c}
         \vdots & \vdots & \vdots & \vdots\\
         z^{[1](1)} & z^{[1](2)} & \cdots & z^{[1](m)}\\
         \vdots & \vdots & \vdots & \vdots\\
         \end{array}
         \right]
   $$
   $$
   A^{[1]} =
      \left[
         \begin{array}{c}
         \vdots & \vdots & \vdots & \vdots\\
         \alpha^{[1](1)} & \alpha^{[1](2)} & \cdots & \alpha^{[1](m)}\\
         \vdots & \vdots & \vdots & \vdots\\
         \end{array}
         \right]
   $$
   $$
         \begin{cases}
         \text{$Z^{[1]} = W^{[1]}X+b^{[1]}$}\\
         \text{$A^{[1]} = \sigma(z^{[1]})$}\\
         \text{$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$}\\ 
         \text{$A^{[2]} = \sigma(z^{[2]})$}\\ 
         \end{cases}
   $$
   从水平上看，矩阵$A​$代表了各个训练样本。从竖直上看，矩阵$A​$的不同的索引对应于不同的隐藏单元。

   对于矩阵$Z，X$情况也类似，水平方向上，对应于不同的训练样本；竖直方向上，对应不同的输入特征，而这就是神经网络输入层中各个节点。

5. 激活函数
   1. sigmod函数：$a = \sigma(z) = \frac{1}{{1 + e}^{- z}}$
      - 使用场合：除了输出层是一个二分类问题基本不会用它
   2. tanh函数：$a= tanh(z) = \frac{e^{z} - e^{- z}}{e^{z} + e^{- z}}$，穿过了$(0,0)$点，并且值域介于+1和-1之间。
      - 使用场合：几乎适合所有场合
      - sigmoid函数和tanh函数两者共同的缺点：在$z$特别大或者特别小的情况下，导数的梯度或者函数的斜率会变得特别小，最后就会接近于0，导致降低梯度下降的速度
   3. Relu函数：$a=max(0,z)$
      - 使用场合：最常用的默认函数
   4. Leaky Relu函数：$a=max(0.01z,z)$
      - Relu函数的优点：
        1. 在z​z​的区间变动很大的情况下，激活函数的导数或者激活函数的斜率都会远大于0
        2. Relu和Leaky ReLu函数大于0部分都为常数，不会产生梯度弥散现象

   ![w600](images/L1_week3_9.jpg)
6. 激活函数的导数
   1. sigmoid：
   
      $$g(z) = \frac{1}{1 + e^{-z}}$$
      
      $$\frac{d}{dz}g(z) = {\frac{1}{1 + e^{-z}} (1-\frac{1}{1 + e^{-z}})}=g(z)(1-g(z))$$
   2. tanh：
   
      $$g(z) = tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$

      $$\frac{d}{{d}z}g(z) = 1 - (tanh(z))^{2}$$
   3. Relu:
   
      $$g(z) =max (0,z)$$

      $$
      g(z)^{'}=
      \begin{cases}
      0&	\text{if z < 0}\\
      1&	\text{if z > 0}\\
      undefined&	\text{if z = 0}
      \end{cases}
      $$
   4. Leaky ReLU:
      $$
      g(z)=\max(0.01z,z) \\
         \\
         \\
      g(z)^{'}=
      \begin{cases}
      0.01& 	\text{if z < 0}\\
      1&	\text{if z > 0}\\
      undefined&	\text{if z = 0}
      \end{cases}
      $$
7. 神经网络的梯度下降法
   1. **Cost function**:
   $$J(W^{[1]},b^{[1]},W^{[2]},b^{[2]}) = {\frac{1}{m}}\sum_{i=1}^mL(\hat{y}, y)$$
   2. **Gradiant descent**:
   $$dW^{[1]} = \frac{dJ}{dW^{[1]}},db^{[1]} = \frac{dJ}{db^{[1]}}$$

   $${d}W^{[2]} = \frac{{dJ}}{dW^{[2]}},{d}b^{[2]} = \frac{dJ}{db^{[2]}}$$
   3. **forward propagation**:
      1. $z^{[1]} = W^{[1]}x + b^{[1]}$
      2. $a^{[1]} = \sigma(z^{[1]})$
      3. $z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$
      4. $a^{[2]} = g^{[2]}(z^{[z]}) = \sigma(z^{[2]})$
   4. **back propagation**：
      1. $dz^{[2]} = A^{[2]} - Y , Y = \begin{bmatrix}y^{[1]} & y^{[2]} & \cdots & y^{[m]}\\ \end{bmatrix}$
      2. $dW^{[2]} = {\frac{1}{m}}dz^{[2]}A^{[1]T}$
      3. ${\rm d}b^{[2]} = {\frac{1}{m}}np.sum({d}z^{[2]},axis=1,keepdims=True)$
      4. $dz^{[1]} = \underbrace{W^{[2]T}{\rm d}z^{[2]}}_{(n^{[1]},m)}\quad*\underbrace{{g^{[1]}}^{'}}_{activation \; function \; of \; hidden \; layer}*\quad\underbrace{(z^{[1]})}_{(n^{[1]},m)}$
      ![](images\dz1prove.png)
      5. $dW^{[1]} = {\frac{1}{m}}dz^{[1]}x^{T}$
      6. ${\underbrace{db^{[1]}}_{(n^{[1]},1)}} = {\frac{1}{m}}np.sum(dz^{[1]},axis=1,keepdims=True)$
8. 随机初始化

   对于一个神经网络，如果你把权重或者参数都初始化为0，导致隐藏层中每一个单元相同(隐含单元对称)，那么梯度下降将不会起作用

   正确的随机初始化：
   1. $W^{[1]} = np.random.randn(2,2)\;*\;0.01$
   2. $b^{[1]} = np.zeros((2,1))$
   3. $W^{[2]} = np.random.randn(2,2)\;*\;0.01$
   4. $b^{[2]} = 0$
   > 乘以常数0.01的原因：当使用用tanh或者sigmoid激活函数，如果$W$很大，$z$就会很大或者很小，导致梯度下降减缓，学习率变慢