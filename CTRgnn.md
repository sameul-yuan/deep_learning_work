# CTR预估模型
- 传统方法
    - LR,POLY2， FM,FFM,GBDT+LR, FTRL, LS-PLM（large-scale piece-wise linear model)等
    - 特征工程
    - 解释性强，训练部署方便，可在线学习
    - 模型校准： 线上和线下的平均CTR相同
    - 内存优化:
        - 稀疏解
        - 特征优化，罕见特征一一定概率加入模型
        - 量化精度
        - 多模型并行训练，共享训练数据，单独保存参数
- 深度学习方法
    - DSSM (deep structrued semantic Model)->搜索广告
        - 基于监query和document对的点击规律监督学习query和document的低维语义表示
        - LSA:是通过单词的document-word的矩阵SVD降维学习
        - Word-hash: 每个word根据其char-level的n-gram来表示
            - 不同的单词形态在char-level空间接近
            - 缓解OOV(out of vocaburary)问题
    - FNN：FM预训练+DNN微调
    - PNN：product层捕获离散特征的高阶组合
        - IPNN： 第一个隐藏层先对一阶特征和内积交叉特征进行类似卷积核的映射变换
        - OPNN： 第一个隐藏层先对一阶特征和外积交叉特征进行类似卷积核的映射变换
        - 内积可视为一系列的AND&&OR操作，外积可可视为一系列AND操作，product层可以看做学习AND/OR规则

    - DeepCrossing：输入原始特征embedding进行拼接+DNN+ResNet（串行）
    - Deep&Cross(DCN)并行: 
        - 在每层进行显示的特征交叉(cross network)
        - cross network + DNN
        - cross network 参数量为2*Lc*d (d为输入x0的维度)，远低于DNN
       
    - wide&deep: 广义线性模型(人工特征交叉)+DNN模型联合训练
        - 离散特征编码+丢弃出现次数太低的特征值
        - 连续特征归一化到0~1
            1. 累积分布函数归一化
                >  $f(x)=P(X \le x)=\frac{\sum_{i=1}^{N}.I(X_i\le x)}{N}$
            2. 将$f(x)$映射到`q`分位
        - wide部分每次有新数据到到是会warm-up重新训练，wide部分和DNN部分开输入
        - model-serving阶段为满足10ms响应，开启多线程优化 

    - DeepFM
    - xDeepFM：
        - CIN(compress interaction network)显示的特征交织(vector wise交互)
        - CIN的池化输出沿着embedding的维度
        - CIN网络深度为1，卷积核参数为1时退化为DeepFM模型
        - 论文中CIN的最好深度为3，最好激活函数为identity，cin隐层维度为100~200
    - NFM
        - FM的提升版本，利用Bi-Interaction进行二阶特征交叉并输出DNN
        - BI-interaction
            > $f_{BI}(\mathbb{V_x}) = \sum_{i=1}^{n}\sum_{j=i+1}^{n}x_i\mathbb{v_i}\odot x_j\mathbb{v_j}$  
            $\odot$表示逐元素乘法，$f_{BI}$为一个`k`维向量
        - 模型输出
            > $y_{NFM}(\mathbb{x}) = w_o+\mathbb{w.x} + f(\mathbb{x})$  
            $f(\mathbb{x}) = \mathbb{w_l*h_l}$为DNN最后隐藏层输出   
            如果没有隐藏层且 $\mathbb{w_l}=[1,1,1,1,\cdots]$则是标准的FM
        - 将Bi-Interaction替换为拼接层则退化为wide&deep 模型
        - 将Bi-interAction输入DNN相当于给神经网络输入了有效的二阶交互信息减轻了DNN的学习负担，因此DNN只有一个隐层时表现就已经很好；
        - 将DeepFM的DNN部分换成Bi-Interaction+DNN效果会怎么样？

    - AFM
        - 模型结构和NFM基本类似，但缺少隐藏层提取高阶特征
        - 将NFM的Bi-Interaction 换成pair-wise interaction
            - pair-wise interaction得到n(n-1)/2个向量
            - 将n(n-1)/2个向量进行sum-pooling即为Bi-interaction的结果
            - 然而，AFM认为交叉向量的重要性并不一样， 对这n(n-1)/2个向量进行attention pooling，
                >$f_{PI}(\mathbb{V_x}) = \sum_{i=1}^{n}\sum_{j=i+1}^{n}a_{ij}*x_i\mathbb{v_i}\odot x_j\mathbb{v_j}$ 
            - attention的因子通过attention network来学（全连接网络)
                > $\hat{a}_{ij} = \mathbb{h}*relu (\mathbf{W}(x_i \mathbb{v_i} \odot x_j\mathbb{v_j})+\mathbb{b})$  
                $a_{ij}= softmax(\hat{a}_{ij})$
        - AFM pair-wise Interaction 和NFM Bi-Interaction的输出都有dropout
        - 网格搜索超参
        - 加上DNN会怎么样？

    - ESMM: CVR（点击转化率预估)
        - 通过多任务同时训练CVR和CTR来解决样本选择偏差问题(CVR在点击样本训练，但是推断是在整个曝光样本)
        - p(z=1|y=1, x) = p(z=1,y=1|x)/p(y=1|x)

    - DIN:(Deep Interest network)：展示广告
        - 用户历史行为数据建模
        - 普通DNN对历史行为emdedding进行笼统的sum-pooling/avg-pooling会丢失很多信息
        - DIN通过对不同的广告有不同的user embedding
            - 历史行为emdedding 的 attention 加权和
            - 权重右每个行为的embedding 和广告的embedding计算
        - Dice激活函数： 依赖数据mini-batch的均值和方差，通过数据调整整流点，pReLU只在一个点；
        - 自适应正则化： 低频特征对应权重的正则化系数更大
    - DIEN:(Deep intersest Evolution network)
        - interest extract layer: 从行为序列捕获潜在的时序兴趣（GRU）
        - interest evolution layer：建模用户的兴趣演变，基于兴趣抽取层GRU输出和广告embedding 计算attention score作为兴趣进化层GRU的update_gate
    - DSIN:(Deep session interest network)
        - 该模型利用用户的历史会话来建模用户行为序列
        - 引入了multi-header self-attention 进行会话兴趣提取
        - 引入BI_LSTM 进行不同会话的兴趣交互
        - ad的embedding和会话兴趣、兴趣交互的输出分别进行attention激活级联其他特征送入FC

    - DICM:(deep Image CTR model)
        - 引入用户的行为图片建模，利用视觉偏好增强用户的行为表示
    




    

         



        

         


    
    



# 图网络与推荐系统
GNN的限制：
- 容易受到攻击： 修改节点属性，增删边
- 对图的区别能力： 不同层聚合需要满足injective neighbor aggegation（一一映射), injective multi-set function: $\Phi(\sum_{x \in S}f(x))$
## session-based recommendation with gnn  
1. 在一个session中只有session中有限的用户历史行为序列
2. session的行为序列构成一副子图，不同的session构成一个全局图
3. session embedding = session中最后一个item的embedding || attention embedding of items in the session
4. 训练集的构造： 序列中的从头开始截取不同长度的序列

## graph neural network for social recommendation 
1. 两类图网络
    - user-modeling
        - user-item graph 
            - item embedding($q_i$) , opinion embedding($r_i$)
            - item的中间表示$a_{i}$= MLP($q_i$||$r_i$)
            - 经过attention(MLP($p_i$||$a_i$)交互items的$a_{i}$学习user 在item-space中的latent表示$h^I$
        - user-user social graph
            - user embedding($p_{i}$)
            - 交互用在item-space中的表示$h^I$
            - 经过attention(MLP($p_i$||$h_I$)交互user的$h^I$学习user在social-space中的latent 表示$h^S$
        - MLP 融合$h^I$和$h^S$来表示用户的最终latent factor $h_i$
    - item-modeling
        - user_item graph
            - interacted users and there opinions related to  item
            - 学习opinion-aware 的用户中间表示$f_u$=MLP($p_i$||$r_i$)
            - attention(MLP($q_i$||$f_u$))交互users的$f_u$来比表示item的最终latent factor $z_j$
    - rating prdiction
    $$
    g_l = MLP(h_i||z_i),\quad r_{ij} = \bold{w^T}g_l
    $$

    - loss
    $$
    Loss = \frac{1}{2|O|}\sum_{i,j\in O}\left(r_{ij}^{'}-r_{ij}\right)^2
    $$
    - ablation analysis
        - social network : 重要
        - attention： 非常重要，尤其是用户关系强弱
        - embedding size： 先升(8-64)后降(>64)

    - 后续：
        - user 和item和其他边信息
        - rating和关系的冬天变化

