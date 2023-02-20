## 第三周 超参数调试、Batch正则化和程序框架
1. 调试处理
   > 在超参数中，$a$无疑是最重要的
   
   超参数调试方法：
   1. 随机取值(图右)

   ![](images\75bfa084ea64d99b1d01a393a7c988a6.png)

   之所以这么做是因为，对于你要解决的问题而言，你很难提前知道哪个超参数最重要，正如你之前看到的，一些超参数的确要比其它的更重要。

   2. 精确搜索

   ![](images\c3b248ac8ca2cf646d5b705270e01e78.png)

   当发现效果最好的某个点，那在接下来要做的是放大这块小区域（小蓝色方框内），然后在其中更密集得取值或随机取值；在更小的方格中，你可以更密集得取点。所以这种从粗到细的搜索也经常使用。
2. 为超参数选择合适的范围

   用**对数标尺**搜索超参数的方式会更合理：

   1. 假设你在搜索超参数$\alpha$（学习速率），如果你在$10^{a}$和$10^{b}$之间取值。你要做的就是在$[a,b]$区间随机均匀地给$r$取值，这个例子中$r \in \lbrack - 4,0\rbrack$，然后你可以设置$\alpha$的值，基于随机取样的超参数$\alpha =10^{r}$

      ![](images\a54d5ea6cd623f741f75e62195f072ca.png)
      ```py
      r = -4 * np.random.rand() #-4<=r<=0
      a = 10**r
      ```
   2. 另一个例子是给$\beta$ 取值，范围为0.9到0.999之间
      
      ![](images\2e3b1803ab468a94a4cae13e89217704.png)

      考虑这个问题最好的方法就是，在0.1到0.001区间内给$1-\beta$取值
      ```py
      r = -2 * np.random.rand()-1 #-3<=r<=-1
      b = 1 - 10**r
      ```
3. 超参数调试的实践：Pandas VS Caviar
   
   ![](images\a361c621a9a0a1a99b03eef8716c5799.png)

   1. Panda: 一个模型，实验中改良
   2. Caviar: 多种模型同时学习

   这两种方式的选择，是由你拥有的计算资源决定的：当拥有足够的计算机去平行试验许多模型，那采用Caviar；当数据太多或没有许多计算资源或足够的CPU和GPU，采用Panda

4. Batch归一化 (Batch Norm)
   > Batch归一化会使你的参数搜索问题变得很容易，使神经网络对超参数的选择更加稳定，超参数的范围会更加庞大，工作效果也很好，也会是你的训练更加容易，甚至是深层网络。
   - 公式：
   $$\mu = \frac{1}{m}\sum_iz^{(i)}$$
   $$\sigma^2 = \frac{1}{m}\sum_i(z^{(i)}-\mu)^2$$
   $$z_{norm}^{(i)} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}$$
   $$\tilde{z}^{(i)} = \gamma z_{norm}^{(i)}+\beta$$
   - 作用：它适用的归一化过程，不只是输入层，甚至同样适用于神经网络中的深度隐藏层
   - 通过赋予$\gamma$和$\beta$值，可以构造含指定平均值和方差的隐藏单元值；如果$\gamma= \sqrt{\sigma^{2} +\varepsilon}$，$\beta = \mu$，则${\tilde{z}}^{(i)} = z^{(i)}$，即$\tilde{z}^{(i)}$还是正态分布$(\mu=0, \sigma^2=1)$
   - $\gamma$和$\beta$的维数：($n^{[l]},1$)
   - 在使用**Batch**归一化，其实你可以消除这个参数（$b^{[l]}$），因为**Batch**归一化超过了此层$z^{[l]}$的均值，$b^{[l]}$这个参数没有意义
   - 如何用**Batch**归一化来应用梯度下降法:

     for $t=1$ to **num MiniBatch**
     
        1. 在$X^{\left\{ t\right\}}$上计算正向**prop**，每个隐藏层都应用正向**prop**，用**Batch**归一化代替$z^{[l]}$为${\tilde{z}}^{[l]}$。
        2. 用反向**prop**计算$dw^{[l]}$和$db^{[l]}$，$d{\beta}^{[l]}$和$d\gamma^{[l]}$。
        3. 更新这些参数：$w^{[l]} = w^{[l]} -\text{αd}w^{[l]}$，和以前一样，${\beta}^{[l]} = {\beta}^{[l]} - {αd}{\beta}^{[l]}$，对于$\gamma$也是如此$\gamma^{[l]} = \gamma^{[l]} -{αd}\gamma^{[l]}$。（也可以用momentum，RMSpop，Adam等算法）
     ![](images\797be77383bc6b34ddd2ea9e49688cf6.png)
   - 原理：
     1. 对于这里的输入值，和隐藏单元的值归一化，以此来加速学习
     2. 它减弱了前层参数的作用与后层参数的作用之间的联系，它使得网络每层都可以自己学习，稍稍独立于其它层，这有助于加速整个网络的学习。
     3. 有轻微的正则化效果(Batch Norm的副作用)，min-batch的数量越大，正则化效果越弱
   - 测试时的 Batch Norm：

     在训练时，$\mu$和$\sigma^{2}$是在整个**mini-batch**上计算出来的包含了像是64或328或其它一定数量的样本，但在测试时，你可能需要逐一处理样本，方法是根据你的训练集估算$\mu$和$\sigma^{2}$，我们通常运用指数加权平均来追踪在训练过程中你看到的$\mu$和$\sigma^{2}$的值。
5. Softmax regression：一个用于多分类的算法
   
   ![](images\08e51a30d97c877410fed7f7dbe1203f.png)
   1. Output function：$\hat{y}=g{{({{w}^{T}}{{x}^{(i)}}+b)}}$, where $g{(z)} = \frac{e^{(z)}}{\sum_{j=1}^{C}e^{(z)}_j}$
   2. Loss function：$L(\hat y,y ) = - \sum_{j = 1}^{C}{y_{j}log\hat y_{j}}$
   3. Cost function：$J( w^{[1]},b^{[1]},\ldots\ldots) = \frac{1}{m}\sum_{i = 1}^{m}{L( \hat y^{(i)},y^{(i)})}$
   4. Backward propagation：$dz^{[l]} = \hat{y} -y$
   5. 例：
   
      ![](images\b433ed42cdde6c732820c57eebfb85f7.png)

      ![](images\ed6ccb8dc9e65953f383a3bb774e8f53.png)
6. Tensorflow例子：
   - 要求：使用tensorflow将$J(w)= w^{2}-10w+25$最小化
   1. 引入tensorflow：
      ```py
      import tensorflow as tf
      ```
   2. 系数:
      ```py
      w = tf.Variable(0, dtype=tf.float32)
      ```
   3. 训练数据x（之后会赋值）:
      ```py
      x = tf.placeholder(tf.float32, [3,1])
      ```
   4. 损失函数
      ```py
      cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
      ```
   5. 定义train为学习算法
      ```py
      train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
      ```
   6. 开启一个TensorFlow Session，初始化并评估w
      ```py
      init = tf.global_variables_initializer()
      session = tf.Session() 
      session.run(init)
      print(session.run(w))
      ```
      也可以是：
      ```py
      with tf.Session() as session:
          session.run(init)
          print(session.run(w))
      ```
   7. 运行梯度下降1000次迭代后后再评估w
      ```py
      for i in range(1000):
         session.run(train, feed_dict = {x:coefficient})
      print(session.run(w))
      ```