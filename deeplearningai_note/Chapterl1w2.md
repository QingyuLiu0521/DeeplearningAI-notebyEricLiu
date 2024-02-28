## 第二周：神经网络的编程基础(Basics of Neural Network programming


1. **符号定义** ：

    $x$：表示一个$n_x$维数据，为输入数据，维度为$(n_x,1)$； 

    $y​$：表示输出结果，取值为$(0,1)​$；

    $(x^{(i)},y^{(i)})$：表示第$i$组数据，可能是训练数据，也可能是测试数据，此处默认为训练数据； 

    $X=[x^{(1)},x^{(2)},...,x^{(m)}]$：表示所有的训练数据集的输入值，放在一个 $n_x×m$的矩阵中，其中$m$表示样本数目; 

    $Y=[y^{(1)},y^{(2)},...,y^{(m)}]$：对应表示所有训练数据集的输出值，维度为$1×m$。

    用一对$(x,y)$来表示一个单独的样本，$x$代表$n_x$维的特征向量，$y$ 表示标签(输出结果)只能为0或1。而训练集将由$m$个训练样本组成，其中$(x^{(1)},y^{(1)})$表示第一个样本的输入和输出，$(x^{(2)},y^{(2)})$表示第二个样本的输入和输出，直到最后一个样本$(x^{(m)},y^{(m)})$，然后所有的这些一起表示整个训练集。有时候为了强调这是训练样本的个数，会写作$M_{train}$，
2. `X.shape`(Python指令)：用于显示矩阵的规模。例：`X.shape`等于$(n_x,m)$
3. Logistic regression(逻辑回归)：一个用于二分类(**binary classification**)的算法
   1. $\hat{y}$：训练集的预测值
   2. $y$：训练集的实际值
   3. $w、b$：参数
   4. $m$：样本数
   5. Output function: $\hat{y}=\sigma{({{w}^{T}}{{x}^{(i)}}+b)}$, where $\sigma{(z)} = \frac{1}{1+e^{-z}}$
   6. Loss function：$$L\left( \hat{y},y \right)=-y\log(\hat{y})-(1-y)\log (1-\hat{y})$$损失函数衡量算法在单个训练样本中表现，**通过训练改变参数$w、b$进而改变预测值$\hat{y}$，让损失函数的值越小越好**
   7. Cost function:$$J\left( w,b \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{L\left( {{{\hat{y}}}^{(i)}},{{y}^{(i)}} \right)}=\frac{1}{m}\sum\limits_{i=1}^{m}{\left( -{{y}^{(i)}}\log {{{\hat{y}}}^{(i)}}-(1-{{y}^{(i)}})\log (1-{{{\hat{y}}}^{(i)}}) \right)}$$代价函数衡量算法在全部训练样本上的表现；通过训练代价函数最终得到参数$w$和参数$b$
4. Gradiant descent algorithm(梯度下降法)
   
   ![](images/c5eda5608fd2f4d846559ed8e89ed33c.jpg)
   > $J(w,b)$是一个凸函数(**convex function**)
   1. 初始化(对于逻辑回归几乎所有的初始化方法都有效)
   2. 朝最陡的下坡方向走一步，不断地迭代
      1. $:=$表示更新参数
      2. $a$ 表示学习率（**learning rate**），用来控制步长（**step**），即向下走一步的长度
      3. 不断重复下列操作：
         1. $w:=w-\alpha{\frac{\partial J(w,b)}{\partial w}}$
         2. $b:=b-\alpha{\frac{\partial J(w,b)}{\partial b}}$
   3. 直到走到全局最优解或者接近全局最优解的地方
5. 计算图（Computation Graph）
   1. 正向计算(从左到右)来计算成本函数J
   2. 反向计算(从右到左)导数

      ![](images/cd75ffa2793fa4af02bdd869fe962bc1.png)
6. 逻辑回归中的梯度下降（Logistic Regression Gradient Descent）
   
   ![](/images/6403f00e5844c3100f4aa9ff043e2319.jpg)
   1. ${da} = \frac{{dL}(a,y)}{{da}}  =  - \frac{y}{a} + \frac{(1 - y)}{(1 - a)}$
   2. ${dz} = \frac{{dL}(a,y)}{{dz}} = \frac{{dL}}{{dz}} = \left( \frac{{dL}}{{da}} \right) \cdot \left(\frac{{da}}{{dz}} \right) = ( - \frac{y}{a} + \frac{(1 - y)}{(1 - a)})\cdot a(1 - a) = a - y$
   3. $dw_1 = \frac{\partial L(w,b)}{\partial w_1} = x_1 \cdot dz$
   4. $db = \frac{\partial L(w,b)}{\partial b} = dz$
7. m个样本的梯度下降(Gradient Descent on m Examples)
   
   初始化$J=0,d{{w}_{1}}=0,d{{w}_{2}}=0,db=0$

   伪代码：
   ```
   J=0;dw1=0;dw2=0;db=0;
   for i = 1 to m
       z(i) = wx(i)+b;
       a(i) = sigmoid(z(i));
       J += -[y(i)log(a(i))+(1-y(i)）log(1-a(i));
       dz(i) = a(i)-y(i);
       dw1 += x1(i)dz(i);
       dw2 += x2(i)dz(i);
       db += dz(i);
   J/= m;
   dw1/= m;
   dw2/= m;
   db/= m;
   w=w-alpha*dw
   b=b-alpha*db
   ```
   我们的目标是不使用**for**循环，而是向量，我们可以这么做：

   $Z = w^{T}X + b = np.dot( w.T,X)+b$

   $A = \sigma( Z )$

   $dZ = A - Y$

   ${{dw} = \frac{1}{m}*X*dz^{T}\ }$

   $db= \frac{1}{m}*np.sum( dZ)​$

   $w: = w - a*dw$

   $b: = b - a*db$
8. 向量化
   
   在深度学习中，向量化代替for循环来处理大数据集，相比for有更高的效率
   ```py
   import numpy as np #导入numpy库
   a = np.array([1,2,3,4]) #创建一个数据a
   print(a)
   # [1 2 3 4]

   import time #导入时间库
   a = np.random.rand(1000000)
   b = np.random.rand(1000000) #通过round随机得到两个一百万维度的数组
   tic = time.time() #现在测量一下当前时间

   #向量化的版本
   c = np.dot(a,b)
   toc = time.time()
   print("Vectorized version:" + str(1000*(toc-tic)) +"ms") #打印一下向量化的版本的时间

   #继续增加非向量化的版本
   c = 0
   tic = time.time()
   for i in range(1000000):
      c += a[i]*b[i]
   toc = time.time()
   print(c)
   print("For loop:" + str(1000*(toc-tic)) + "ms")#打印for循环的版本的时间
   ```

   结果
   ```
   [1 2 3 4]
   Vectorized version:0.9989738464355469ms
   250194.36404996118
   For loop:386.96837425231934ms
   ```
   > np.dot()函数主要有两个功能，向量点积和矩阵乘法

9. 向量化logistics回归

   利用 $m$ 个训练样本一次性计算出小写 $z$ 和小写 $a$，用一行代码即可完成。
   ```
   Z = np.dot(w.T,X) + b
   ```

   ![](images/3a8a0c9ed33cd6c033103e35c26eeeb7.png)
   > A.T: 矩阵A转置
10. 广播(broadcast)
    
    numpy 对不同形状(shape)的数组进行数值计算的方式
    ```
    import numpy as np
    a = np.array([[1,2,3,4],[5,6,7,8]]).T
    print(a)
    print(a/[10,20])
    ```
    答案：
    ```
    [[1 5]
     [2 6]
     [3 7]
     [4 8]]
    [[0.1  0.25]
     [0.2  0.3 ]
     [0.3  0.35]
     [0.4  0.4 ]]
    ```

11. **Python**中的**numpy**一维数组的特性

    建议你编写神经网络时，不要使用shape为 _(5,)_、_(n,)_ 或者其他一维数组的数据结构。

    ```
    # row vector
    a = np.random.randn(1, 5)
    # column vector
    b = np.random.randn(5, 1)
    ```
