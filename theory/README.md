# Interview





## 1. 数学基础

### L1不可导的时候怎么办

损失函数不可导=>~~gradient descent~~, 坐标轴下降法

m个features, 固定m-1个值, 求另一个的局部最优解

Proximal Algorithm 优化损失函数上界的结果

### 讲一下PCA

线性降维方法

新特征的方差尽量大 

具体计算: 中心化操作, 计算协方差矩阵, 特征值分解, 取最大的n个特征值对应的特征向量构造投影矩阵



### 牛顿法

求解无约束最优化问题



## 2. Machine Learning Algorithm

### 交叉熵公式

衡量2个概率分布距离





## 3. Deep Learning

### ReLU vs sigmoid

+ ReLU导数计算更快
+ 没有梯度消失问题