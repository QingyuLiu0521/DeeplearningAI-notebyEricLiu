## 第二周 深度卷积网络：实例探究
1. 经典网络
   1. LeNet-5

      ![](images/5e59b38c9b2942a407b49da84677dae9.png)

      模式：
      ```mermaid
      graph LR;
      　　Input-->|conv|Layer1;
      　　Layer1-->|pool|Layer2;
      　　Layer2-->|n*convpool|LayerN;
      　　LayerN-->|fc|Output;
      ```
      相比现代版本，这里得到的神经网络会小一些，只有约6万个参数；在论文中，LeNet-5使用的正是sigmod函数和tanh函数。
   2. AlexNet

      ![](images/92575493ecd20003b0b76ac51de0efbb.png)

      这种神经网络与LeNet有很多相似之处，不过AlexNet要大得多。正如前面讲到的LeNet或LeNet-5大约有6万个参数，而AlexNet包含约6000万个参数，能够处理非常相似的基本构造模块。AlexNet比LeNet表现更为出色的另一个原因是它使用了ReLu激活函数。

   3. VGG
      > **VGG-16**的这个数字16，就是指在这个网络中包含16个卷积层和全连接层

      ![](images/0a29aeae65a311c56675ad8f1fec2824.png)
    
      模式：几个卷积层($n_H、n_W$不变)后面跟着可以压缩图像大小的池化层($n_H、n_W×\frac{1}{2}$);同时，每次池化层后面的卷积层的过滤器数量都会翻倍($n_c^{[l]}=2×n_c^{[l-1]}$)

      它的主要缺点是需要训练的特征数量非常巨大。
2. 残差网络
   
   作用：解决梯度消失和梯度爆炸问题

   跳跃连接（**Skip connection**）：**ResNets**是由残差块（**Residual block**）构建的，即让$a_{[l]}$进行前向反馈进入第二个RELU函数，即：$\ a^{\left\lbrack l + 2 \right\rbrack} = g\left(z^{\left\lbrack l + 2 \right\rbrack} + a^{[l]}\right)$，也就是加上的这个$a^{[l]}$产生了一个残差块。

   ![](images/f0a8471f869d8062ba59598c418da7fb.png)

   如果输入和输出有不同维度，比如输入的维度是128，$a^{\left\lbrack l + 2\right\rbrack}$的维度是256，再增加一个矩阵，这里标记为$W_{s}$，$W_{s}$是一个256×128维度的矩阵，所以$W_{s}a^{\left\lbrack l\right\rbrack}$的维度是256，这个新增项是256维度的向量。你不需要对$W_{s}$做任何操作，它是网络通过学习得到的矩阵或参数，它是一个固定矩阵，**padding**值为0，用0填充$a^{[l]}$，其维度为256，所以者几个表达式都可以。

   ![](images/cefcaece17927e14eb488cb52d99aaef.png)

   作用：对于一个普通网络，随着网络深度的加深，training error会越来越多；而对于ResNets，即使网络再深，训练的表现仍然不错

   ![](images/6077958a616425d76284cecb43c2f458.png)
3. 网络中的网络以及 1×1 卷积
   
   用shape为1×1的过滤器卷积

   ![](images/46698c486da9ae184532d773716c77e9.png)

   作用：保持输入层高度与宽度不变，**压缩通道数量**

   ![](images/49a16fdc10769a86355911f9e324c728.png)
4. Inception 网络

   构建瓶颈层：
   - 作用：显著缩小表示层规模，又不会降低网络性能
   - 例：计算一个5×5过滤器在该模块中的计算成本（上图1.2亿，下图1204万）
  
     ![](images/27894eae037f4fd859d33ebdda1cac9a.png)

     ![](images/7d160f6eab22e4b9544b28b44da686a6.png)

   Inception网络：

   - 基本思想：Inception网络不需要人为决定使用哪个过滤器或者是否需要池化，而是**由网络自行确定这些参数**，你可以给网络添加这些参数的所有可能值，然后把这些输出连接起来，**让网络自己学习它需要什么样的参数，采用哪些过滤器组合**。

   ![](images/16a042a0f2d3866909533d409ff2ce3b.png)

   ![](images/99f8fc7dbe7cd0726f5271aae11b9872.png)

   ![](images/5315d2dbcc3053b0146cabd79304ef1d.png)

   所以Inception网络只是很多这些你学过的模块在不同的位置重复组成的网络
5. 迁移学习（Transfer Learning）
   
   ![](images/7cf0e18b739684106548cbbf0c1dd500.png)
   
   从网上下载一些神经网络开源的实现，包括代码和权重。你可以去掉这个Softmax层，创建你自己的Softmax单元，来实现自己的神经网络。

   当你的任务只有一个很小的数据集，你可以冻结前面的层（不改变参数），取输入图像XX，然后把它映射到这层（softmax的前一层）的激活函数。这样你训练的就是一个很浅的softmax模型，用这个特征向量来做预测。

   如果有一个更大的训练集，应该冻结更少的层，然后训练后面的层；或者你可以直接去掉这几层，换成你自己的隐藏单元和你自己的softmax输出层。如果你有越来越多的数据，你需要冻结的层数越少，你能够训练的层数就越多。
   
   如果你有大量数据，你应该做的就是用开源的网络和它的权重，把这、所有的权重当作初始化，然后训练整个网络。
6. 数据增强（Data augmentation）
   - 数据扩充
     1. Random cropping

        ![](images/709aa552b6a5f4715620047bacf64753.png)
     2. Mirroring
     3. Rotation
     4. Shearing
     5. Local warping
     6. Color shifting
   
        ![](images/d69cfc9648f3a37eede074bd28c74c0d.png)
   - 数据扩充流程
     1. 使用CPU线程，然后它不停的从硬盘中读取数据，所以你有一个从硬盘过来的图片数据流
     2. CPU线程持续加载数据，然后**实现任意失真变形**，从而构成批数据或者最小批数据，这些数据持续的传输给其他线程或者其他的进程
     3. 开始训练，可以在CPU或者GPU上实现训一个大型网络的训练。

     ![](images/5ee17d350497cb8cf52881f14cb0d9e8.png)
7. 计算机视觉现状
   
   学习算法有两种知识来源：
   1. 被标记的数据，就像$(x，y)$应用在监督学习
   2. 手工工程：它可以是源于精心设计的特征，手工精心设计的网络体系结构或者是系统的其他组件

   有助于在基准测试中表现出色的小技巧：
   1. Ensembling(集成)：独立训练几个神经网络，并平均它们的输出
   2. Multi-crop at test time：在多个版本的测试图像上运行分类器并对结果进行平均
   
      ![](images/6027faa79b81f9940281ea36ca901504.png)