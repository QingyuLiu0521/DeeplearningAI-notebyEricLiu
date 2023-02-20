# 第三门课 结构化机器学习项目（Structuring Machine Learning Projects）

## 第一周 机器学习（ML）策略（1）
1. ML策略：

   如何更快速高效地优化机器学习系统
2. 正交化：
   - 正交化是指，用一个“旋钮”只控制一个变量，而不是多个变量
   - Chain of assumptions in ML
     1. 系统在训练集(training set)上能得出不错的结果
     2. 系统在开发集(dev set)上能得出不错的结果
     3. 系统在测试集(testing set)上能得出不错的结果
     4. 系统在实际应用中表现出色
3. 单一数字评估指标
   
   在进行机器学习项目时，有一个单实数评估指标会让进展快得多

   例：
   - 查准率P：分类器中结果为true的比例
   - 查全率R：分类器正确识别的比例
   - F1分数：$\frac{2}{\frac{1}{P} + \frac{1}{R}}$(P和R的调和平均数)

   ![](images\e75299b34b8ccb565a1566d2a515f597.png)
4. 满足和优化指标

   如果要考虑$N$个指标，可以选择一个指标做为优化指标，剩下$N-1$个指标都是满足指标。
   - 满足指标：一个可接受的阈值（约束条件）
   - 例：构建一个系统来检测唤醒语
     - 优化指标：被唤醒的概率(最大化)
     - 满足指标：The num of false positive $\leq1$ per 24 hours
5. 训练/开发/测试集划分
   
   机器学习的目标：开发集+单实数评估指标 (dev + Metric)

   **开发集和测试集必须来自同一分布**
   - 处理方法：将所有数据随机洗牌，放入开发集和测试集
   > 如果dev set和testing set来自不同分布，由于不同分布之间的差异较大，导致最终系统在testing set上表现不佳
6. 开发集和测试集的大小

   当操作规模大得多的数据集时，98%作为训练集，1%开发集，1%测试集更合理
7. 什么时候该改变开发/测试集和指标？
   
   如果你的评估指标无法正确评估好算法的排名，那么就需要花时间定义一个新的评估指标，能够更加符合你的偏好，定义出实际更适合的算法。
   - 例：
     
     ![](images\f8313f1a0ca63a71d0eb3b3b42453220.png)

     假设你在构建一个猫分类器,你决定使用的指标是分类错误率。所以算法$A$和$B$分别有3％错误率和5％错误率，所以算法$A$似乎做得更好。

     原分类错误率：$$Error = \frac{1}{m_{{dev}}}\sum_{i = 1}^{m_{{dev}}}{I\{ y_{{pred}}^{(i)} \neq y^{(i)}\}}$$

     算法$A$由于某些原因，把很多色情图像分类成猫了，算法$B$实际上是一个更好的算法，因为它不让任何色情图像通过。

     修改后的分类错误率：(归一化)$$Error = \frac{1}{\sum_{}^{}w^{(i)}}\sum_{i = 1}^{m_{{dev}}}{w^{(i)}I\{ y_{{pred}}^{(i)} \neq y^{(i)}\}}$$
     $$
         w^{(i)}=
         \begin{cases}
         \text{$10$, $x^{(i)}是色情图片$}\\
         \text{$1$，$x^{(i)}不是色情图片$}\\
         \end{cases}
     $$
    - 从上面的例子引出如何处理机器学习问题：
      1. 弄清楚如何定义一个指标来衡量你想做的事情
      2. 单独考虑如何在这个指标上做得好
      
      ![](images\12c7390dfc8b80057483de00d0d2c7b7.png)
8. 为什么是人的表现？

   贝叶斯最优错误率: **理论上可能达到的最优错误率**(从$x$到$y$映射的理论最优函数)
    
   ![](images\f44d03275801ce5ec97503851eb22ad5.png)

   如上图所示，机器学习的进展往往相当快，直到你超越人类的表现之前一直很快，当你超越人类的表现时，有时进展会变慢。
9. 可避免偏差（Avoidable bias）
   
   ![](images\a582a5304dca46ff2f450d934bf7330a.png)

   $$Bayes\,optimal\,error\approx Human\,level\,error $$
   $$Avoidable\,bias = Training\,error - Bayes\,optimal\,error$$
   1. 可避免偏差可以衡量或者估计你的学习算法的bias问题有多严重
   2. 训练错误率和开发错误率之间的差值告诉你方差上的问题有多大
10. 理解人的表现
    
    对人类水平有大概的估计可以让你做出对贝叶斯错误率的估计，这样可以让你更快地作出决定是否应该专注于减少算法的偏差，或者减少算法的方差。
    
    这个决策技巧通常很有效，直到你的系统性能开始超越人类，那么你对贝叶斯错误率的估计就不再准确了，但这些技巧还是可以帮你做出明确的决定。
11. 超过人的表现
    
    当你的算法超过人的表现(NLP)，现有的一些工具帮助你指明方向的工具就没那么好用了
12. 改善你的模型的表现
    1. 监督学习算法的两个基本假设：
       - 训练集的拟合很好
       - 开发集和测试集表现也很好
    2. Reducing avoidable bias and variance:
       - The revolution of **High avoidable Bias**:
         1. Trainer bigger network (more layers or more units)
         2. Train longer
         3. Use a better optimization algorithms(Momentom, RMSprop, Adam...)
         4. NN architecture/hyperparameters search(CNN, RNN...)
       - The revolution of **High Variance**:
         1. More data
         2. Regularization(L2, dropout...)
         3. NN architecture/hyperparameters search(CNN, RNN...)