## 第二周：优化算法
1. 符号：
   - $X^{\{t\}}$: 第t个mini-batch
2. Mini-batch 梯度下降：
   1. 定义：把训练集分割为小一点的子集训练，这些子集叫**Mini-batch**
     ![](images\112c45cf393d896833ffce29e14fe8bc.png)
     使用batch梯度下降法，一次遍历训练集只能让你做一个梯度下降，使用mini-batch梯度下降法，一次遍历训练集，能让你做$\frac{m}{m_{mini}}$个梯度下降
     > 若$m_{mini}=1$，被称为**随机梯度下降法(SGD)**
   2. **使用mini-batch梯度下降法，如果你作出成本函数在整个过程中的图，则并不是每次迭代都是下降的，会有噪音**，这是因为也许$X^{\{1\}}$和$Y^{\{1\}}$是比较容易计算的**mini-batch**，因此成本会低一些。不过也许出于偶然，$X^{\{2\}}$和$Y^{\{2\}}$是比较难运算的**mini-batch**，或许需要一些残缺的样本，这样一来，成本会更高一些，所以才会出现这些摆动。
   ![](images\b5c07d7dec7e54bed73cdcd43e79452d.png)
   3. 实践中最好选择不大不小的mini-batch尺寸，因为
      1. 得到了大量向量化，比一次性处理多个样本快得多
      2. 不需要等待整个训练集被处理完就可以开始进行后续工作，加快训练速度
   4. **mini-batch**的大小选择：
      1. 当$m>2000$，使用batch梯度下降法
      2. 一般mini-batch大小是$2^n$，64到512的mini-batch大小比较常见。
   5. 具体步骤：
      - Shuffle：如下所示，创建训练集（X，Y）的随机打乱版本。X和Y中的每一列代表一个训练示例。注意，随机打乱是在X和Y之间同步完成的。这样，在随机打乱之后，X的列就是对应于Y中标签的示例。打乱步骤可确保该示例将随机分为不同小批。
        ![](images\kiank_shuffle.png)
        ```py
         # Step 1: Shuffle (X, Y)
         permutation = list(np.random.permutation(m))
         shuffled_X = X[:, permutation]
         shuffled_Y = Y[:, permutation].reshape((1,m))
        ```
      - Partition：将打乱后的（X，Y）划分为大小为mini_batch_size（此处为64）的小批处理。请注意，训练示例的数量并不总是可以被mini_batch_size整除。最后的小批量可能较小，但是你不必担心，当最终的迷你批处理小于完整的mini_batch_size时，它将如下图所示：
        ![](images\kiank_partition.png)
        ```py
         # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
         num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
         for k in range(0, num_complete_minibatches):
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

         # Handling the end case (last mini-batch < mini_batch_size)
         if m % mini_batch_size != 0:
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        ```
3. 指数加权平均：
   $$v_0 = 0$$
   $${{v}_{t}}=\beta {{v}_{t-1}}+(1-\beta ){{\theta }_{t}}$$ 
   例：使用某年伦敦一年的温度作图（红：$\beta=0.9$, 绿：$\beta=0.98$）

   ![](images\a3b26bbce9cd3d0decba5aa8b26af035.png)
   
   当$\beta = 0.9$, $$v_{100} = 0.1\theta_{100} + 0.1 \times 0.9 \theta_{99} + 0.1 \times {(0.9)}^{2}\theta_{98} + 0.1 \times {(0.9)}^{3}\theta_{97} + 0.1 \times {(0.9)}^{4}\theta_{96} + \ldots$$
   $v_t$平均了大约$\frac{1}{(1-\beta)}$天的温度
4. 指数加权平均的偏差修正
   
   当$\beta=0.98$，如果有偏差修正，画出的是绿色曲线；如果没有偏差修正，画出的是紫色曲线，初始时期具有较大偏差

   ![](images\26a3c3022a7f7ae7ba0cd27fc74cbcf6.png)

   偏差修正：(实际不常用)$$\frac{v_{t}}{1- \beta^{t}}$$
5. 动量梯度下降法(Momentum)
   
   ![](images\cc2d415b8ccda9fdaba12c575d4d3c4b.png)

   如图所示，在梯度下降时，会发生摆动；在纵轴上，你希望学习慢一点，因为你不想要这些摆动，但是在横轴上，你希望加快学习，你希望快速从左向右移，移向最小值，移向红点，这时可以用到动量梯度下降法，**来抵消纵向的摇摆**：
   $$v_{dW} = \beta v_{{dW}} + ( 1 - \beta)dW$$
   $$v_{db} = \beta v_{{db}} + ( 1 - \beta)db$$
   $$W:= W -\alpha v_{dW}$$
   $$b:= b -\alpha v_{db}$$
   $$Hyperparameters:\alpha,\beta\,\,\,\,\,\,(\beta = 0.9)$$
6. RMSprop

   加快梯度下降也可以用RMSprop算法

   假设纵轴代表参数$b$，横轴代表参数$W$，所以，你想减缓$b$方向的学习，即纵轴方向，同时加快横轴方向的学习，RMSprop算法可以实现这一点

   $\epsilon$是很小的数字，避免被零除

   ![](images\553ee26f6efd82d9996dec5f77e3f12e.png)
   $$S_{dW} = \beta S_{dW} + ( 1 - \beta )dW^2$$
   $$S_{db} = \beta S_{db} + ( 1 - \beta){db}^2$$
   $$W:= W -\alpha \frac{dW}{\sqrt{S_{dW}}+\epsilon}$$
   $$b:= b -\alpha \frac{db}{\sqrt{S_{db}}+\epsilon}$$
   $$Hyperparameters:\alpha,\beta,\epsilon\,\,\,\,\,\,(\beta = 0.999,\epsilon=10^{-8})$$
7. Adam优化算法
   
   **Adam**优化算法基本上就是将**Momentum**和**RMSprop**结合在一起:
   1. 计算过去梯度的指数加权平均值，并将其存储在变量（使用偏差校正之前）和 （使用偏差校正）中。
   2. 计算过去梯度的平方的指数加权平均值，并将其存储在变量（偏差校正之前）和（偏差校正中）中。
   3. 组合“i”和“ii”的信息，在一个方向上更新参数。

   On iteration t:
   $$v_{dW} = 0，S_{dW} =0，v_{db} = 0，S_{db} =0$$
   $$v_{dW} = \beta_1 v_{{dW}} + (1 - \beta_1)dW$$
   $$v_{db} = \beta_1 v_{{db}} + (1 - \beta_1)db$$
   $$S_{dW} = \beta_2 S_{dW} + (1 - \beta_2)dW^2$$
   $$S_{db} = \beta_2 S_{db} + (1 - \beta_2)db^2$$
   $$v^{corrected}_{dW} = \frac{v_{dW}}{1-\beta_1^t}$$
   $$v^{corrected}_{db} = \frac{v_{db}}{1-\beta_1^t}$$
   $$S^{corrected}_{dW} = \frac{S_{dW}}{1-\beta_2^t}$$
   $$S^{corrected}_{db} = \frac{S_{db}}{1-\beta_2^t}$$
   $$W:= W -\alpha \frac{v^{corrected}_{dW}}{\sqrt{S^{corrected}_{dW}}+\epsilon}$$
   $$b:= b -\alpha \frac{v^{corrected}_{db}}{\sqrt{S^{corrected}_{db}}+\epsilon}$$
   Hyperparameters:
   1. $\alpha$: needs to be tuned
   2. $\beta_1$: 0.9
   3. $\beta_2$: 0.999
   4. $\epsilon$: $10^{-8}$
8. 学习率衰减(Learning rate decay)

   学习率$a$衰减的本质在于，在学习初期，你能承受较大的步伐，但当开始收敛的时候，小一些的学习率能让你步伐小一些。

   学习率衰减的常用公式有如下三种：（**decay-rate**称为衰减率，**epoch-num**为遍历训练集的代数，$\alpha_{0}$为初始学习率）
   1. $a= \frac{1}{1 + decayrate * \text{epoch}\text{-num}}a_{0}$
   2. $a ={0.95}^{\text{epoch-num}} a_{0}$
   3. $a =\frac{k}{\sqrt{\text{epoch-num}}}a_{0}$
9. 局部最优的问题
    
   1. 鞍点：事实上，如果你要创建一个神经网络，通常梯度为零的点并不是这个图中的局部最优点，实际上成本函数的零梯度点，通常是鞍点。

   ![](images\c5e480c51363d55e8d5e43df1eee679b.png)

   2. 平稳段：平稳段会减缓学习，平稳段是一块区域，其中导数长时间接近于0，如果你在此处，梯度会从曲面从从上向下下降，因为梯度等于或接近0，曲面很平坦，你得花上很长时间慢慢抵达平稳段的这个点，因为左边或右边的随机扰动，我们可以沿着这段长坡走，直到这里，然后走出平稳段。

   ![](images\607bd30801c87ed74bb95c49f218f632.png)

   3. 从上面两点你可以知道：
      - 你不太可能困在极差的局部最优中，条件是你在训练较大的神经网络，存在大量参数，并且成本函数JJ被定义在较高的维度空间
      - 平稳段是一个问题，这样使得学习十分缓慢，这也是像Momentum或是RMSprop，Adam这样的算法，能够加速学习算法的地方。在这些情况下，更成熟的优化算法，如Adam算法，能够加快速度，让你尽早往下走出平稳段


$$X(z) = \frac{1-0.5z^{-1}}{1+\frac{3}{4}z^{-1}+\frac{1}{8}z^{-2}}$$