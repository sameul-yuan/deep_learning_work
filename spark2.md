
# 1、spark：快速通用的集群计算平台
- 高级API剥离了对集群本身的关注；
- 内存计算，速度快，支持交互式和流式处理；
- 通用引擎，完成SQL，文本，图网络，机器学习；
### 1.1、基本组成
* 集群管理器(hadoop YARN等)->spark core(计算引擎：完成任务调度分发)->高级组件(sparkSQL, sparkStreaming, sparkMlib等)
* RDD(resilient distributed datasets): spark对分布式数据和计算的基本抽象；

### 1.2、运行架构
- 采用主从结构，驱动器节点->集群管理器->执行器节点
- 驱动器节点： 执行程序中main()方法的进程
	- 把用户程序转为任务， 定义了创建转化执行RDD的DAG图（Directed Acyclic Graph, DAG),并发起在集群上的各种并行操作
	- 通过sparkContext 来访问spark, sparkContext代表了对计算集群的一个连接，通过其sc.stop()来关闭连接
	- 为执行器节点调度任务，基于持续期之前的数据缓存调度，减少传输开销
- 执行器节点
	- 执行具体的spark任务，将结果返回驱动器进程
	- RDD缓存
- 集群管理器
	- 启动执行器节点，甚至驱动器节点
	- spark的可插拔式组件，spark支持多种集群管理器
- 启动程序
	- spark-submit 将应用提交到集群管理器， spark-submit 会帮助引入程序对spark的依赖

### 1.3、使用方式
- 在spark shell中使用, 已经提供了默认的sparkContext, :quit 或 ctrl+D退出
- 在独立程序中链接调用spark：
	- 基于Java的配置好包依赖；
	- 基于python 的通过spark-submit运行脚本
	>
		bin/spark-submit --master spark://host:7077 --excutor-memory 10g my_script.py  
		bin/spark-submit [options] <app jar| python file> [app options]
### 1.4、 打包项目和依赖
1. maven构建
maven 的pom.xml 文件包含了本次构建的定义  
mvn package 构建包
2. sbt 构建
创建build.bst的构建文件  
sbt build.bst

>
	from pyspark import SparkConf, SparkContext
	conf = SparkConf().setMaster('local').setAppName('demo01')
	sc = SparkContext(conf=conf)
	input = 'file:///C:/Users/Samuel/Desktop/data.txt'
	textFile = sc.textFile(input)
	print(textFile.collect())
	wordcnt = textFile.flatMap(lambda x: x.split(' ')).map(lambda(x,1)).reduceByKey(lambda x,y: x+y)
	wordcnt.persist()
	wordcnt.foreach(print)

# 2、spark对数据的所有操作：创建RDD，转化RDD，RDD求值
**RDD概述**: 是不可变的分布式对象集合;分为多个分区，这些分区运行在集群的不同节点上; 可以包含Java,Scala,Python中任意类型对象。

1. 创建RDD 
    - RDD 一旦创建就无法修改
	- 读取外部数据集: `lines = sc.textFile(file)`
	- 程序中的对象集合:`lines=sc.parallelize(someobj)`
	- 惰性求值
2. 转化操作
	- 返回新的RDD, spark会记录RDD的依赖关系以便行动操作和恢复丢失数据
	- 惰性求值
	- map，filter
3. 行动操作
	- 触发实际的计算
	- 返回结果到驱动程序或把结果写入外部系统
	- RDD的每次行动操作都会依据依赖关系重新计算，通过rdd.cache(), rdd.persist()缓存中间结果
		>
			lines = sc.textFile("xxx.md")
			lines.persist() # cache()值缓存在内存, 否则每次都会重新读取
			lines.count() #计算 lines,此时将 lines 缓存
			lines.first() # 使用缓存的 lines
	- count，collect，first，saveAsTextFile
4.  向spark传递可调用对象来支持转化和行动操作
	- spark会将传递的可调用对象分发到各个节点，因此需要确保传递的函数及引用的对象是可序列化的
	- 应该避免将函数的所在的对象也序列化传递出去（传递对象的成员或包含对象的字段），赋给局部变量再传递
	- Python支持lambda表达式，全局函数，局部函数

# 3、常见的创建、转化、行动操作
### 3.1 两种创建方式
- 读取外部数据集
  	>lines = sc.textFile('xxx.md')
- 分发程序中的对象集合
  	>lines = sc.parallelize([1,2,3,4])   #需要将整个数据放到内存中
### 3.2、支持所有类型RDD的转化操作
- `.map(f, preserverPartitioning=False)`: 将f运用在当前RDD的每个元素，构成新的RDD；preserverPartioning=True则新RDD保留旧RDD的分区。
- `.flatMap(f, preserverPartioning=False)`: 将f用于当前RDD的每个元素， 并将返回的迭代器扁平化；可看成map和flatten的级联操作
	>
		lines = sc.parallelize(['hello world','hi'])
		lines.map(lambda line:line.split(" ")) #新的RDD元素为[['hello','world'],['hi',]]    
		lines.flatMap(lambda line:line.split(" ")) #新的RDD元素为 ['hello','word','hi']
- `.mapPartitions(f, preserverPartitoning=False)`:将f用于当前RDD每个分区，返回的迭代器构成新RDD；
	>
		def f(iterator):
			xxx
- `.mapPartitionWithIndex(f, preServerPartioning=False)`:将f用于当前的每个分区及id；	
	>
		def f(index, iterator):
			xxx
- `.filter(f)`:当前RDD中通过f为真的元素构成新RDD
- 伪集合操作
	- `.distinct()`
	- `.union(other)` 不去重
	- `.intersection(other)` 去重，需要通过网络混洗发现重复元素,性能差
	- `.substract(other)` 去重
	- `.catesian(other)` 不去重
- `.sample(replace, frac,seed=None)`
	- replace：True支持重复采样，否则无放回采样
	- frac： replace=True表示每个元素期望选择的次数，replace=False每个元素被期望选择的概率
- `.glom()`: 将RDD中每个分区的元素聚合成一个列表作为新RDD
- `.sortBy(keyfunc, ascending=True, numPartitons=None)`: 对RDD排序，keyfunc是比较函数
- `.groupBy(f, numPartions=None, partionFunc=<xxx>)`:返回分组的RDD
	>
		rdd = sc.parallelize([1, 1, 2, 3, 5, 8])
		result = rdd.groupBy(lambda x: x % 2).collect() #结果为： [(0, [2, 8]), (1[1, 1, 3, 5])]
- `.keyBy(f)`: 创建元素是元祖(f(x),x)的RDD
- `.pipe(command, env=None,checkCode=False)`: 返回外部进程command输出结果构成的RDD
- `.randomSplit(weights,seed=None)`: 旧RDD随机拆分的RDD，列表weights给出了每个结果DataFrame的相对大小
- `.zip(other)` :返回pair RDD, 键来自self而value来自other
- `.zipWithIndex`: 返回pair RDD, 键来自self而value来自值键的index
- `.zipWithUniqueId()`: 返回pair RDD, 键来自self而value来自一个独一无二的ID，不会触发spark job

### 3.3、支持所有类型RDD的行动操作
- 聚合操作
	- `.reduce(f)`: 操作两个相同元素类型的RDD，并返回同类型的新元素
		> sum = rdd.reduce(lambda x,y: x+y) </br>
		> su = rdd.reduce(operator.add)
	- `.fold(zeroValue,op)`: 通过op聚合当前RDD,聚合过程中返回值类型与当前RDD元素类型相同，首先对分区中的每个元素进行聚合（第一个数有zeroValue提供），再将分区聚合结果再次聚合（第一个数由zeroValue提供）
	- `.aggregate(zeroValue, seqOp, combOp)`：聚合当前RDD，不要求返回类型和当前RDD元素类型相同；首先对分区的每个元素用seqOp聚合(第一个数有zeroValue提供)，再将分区聚合结果按照combOp再次聚合(第一个元素由zeroValue提供)
		```scala
		sum_count = nums.aggregate((0,0),
		(lambda acc,value:(acc[0]+value,acc[1]+1),  # seqOp
		(lambda acc1,acc2:(acc1[0]+acc2[0],acc1[1]+acc2[1])) #combOp
		)
		return  sum_count[0]/float(sum_count[1])
		```
- 获取RDD中的元素
	- `.collect()`:返回所有的结果，确保内存能放下
	- `.take(n)`: 以列表的形式返回RDD中的n个元素，取数时会使用尽量少的分区
	- `.takeOrderd(n,key=None)`： 按key指定的顺序返回n个元素，默认降序
	- `.takeSample(replace,num,seed=None)`: 以列表的形式返回RDD随机采样的结果
	- `.top(n,key=None)`: 获取前n个元素
	- `.first()`
- 计数
	- `.count()`
	- `.countByValue()`:字典的形式返回各元素出现的次数
	- `.histogram(buckets)`: buckets指定了分桶方式，为整数则均匀划分的桶数，也可以是指定了分裂点的序列； 返回值为（桶区间序列，桶内元素个数序列)
		>
			rdd = sc.parallelize(range(51))
			rdd.histogram(2)
			# 结果为 ([0, 25, 50], [25, 26])
			rdd.histogram([0, 5, 25, 50])
			#结果为 ([0, 5, 25, 50], [5, 20, 26])
- 统计方法
	- `.max(key=None)`
	- `.mean()`
	- `.min(key=None)`
	- `.sampleStdev()`
	- `.sampleVariance()`
	- `.stdev()`： 与`sampleStdev()`区别是分母是n还是(n-1)
	- `.variance()`
	- `.sum()`
	
- `foreach(f)`: map(f)的行动操作版本
- `foreachPartition(f)`: 
	```scala
	def f(iterator):
		for x in iterator:
			print(x)
    ```
### 3.3、不同类型RDD的转换
- scala中是隐式类型转换
### 3.4、持久化操作
- spark行动操作每次都是重新计算RDD和它的依赖，可对RDD持久化减少计算负担
- `rdd.persist(StorageLevel.DISK_ONLY)` #MEMORY\_ONLY, MEMORY\_ONLY\_SER,MEMORY\_AND\_DISK, MEMORY\_AND_\DISK_SER, DISK_\ONLY
- `rdd.cache()` = `rdd.persist(StorageLevel.MEMORY_ONLY)`
- `rdd.unpersist()`: 移除缓存
- `.getStorageLevel()`: 返回当前的缓存级别

### 3.5 其他方法和属性
- `.context`: 返回RDD 的SparkContext
- `.id()` ：返回RDD的ID
- `.isEmpty()` : 当前仅当RDD为空时返回True
- `.name() `:RDD的名字
- `.setName(name)`: 设置RDD的名字
- `.stats()`: 返回StatCounter对象，计算RDD的统计值
- `.toDebugString()`： 返回RDD的描述字符串用于调试
- `.toLocalIterator()`: 返回对RDD进行遍历的迭代器

# 4、键值对操作
### 4.1 概述
- 键值对RDD通常是一个二元元祖
- 常用语聚合计算
- spark为pair RDD提供了各种并行操作各个键、跨节点重新进行数据分组的接口
### 4.2、pairRDD创建
- 对常规RDD执行转化操作
	>rdd.map(x=>(x.split('')(0), x))
- 键值对格式读取时
- 分发程序中的二元组数据集合
	>rdd = sc.parallelize([('a',1),('b',2)])

### 4.3、pairRDD的转化操作
- 支持所有标准RDD的转换操作，传入的函数参数是二元元祖；
- `.keys()/.values()`
- `.mapValues(f)`: 返回元素为[k, f(v)]的RDD
	> 
		x = sc.parallelize([('a',1),('b',2),('c',3),('d',4)])
		y = x.mapValues(lambda v:v**2)
		print(y.collect) 
		#y: [('a',1),('b',4),('c',9),('d',16)]
	
- `.flatMapValues(f)`: 返回元素为[k, f(v)]的RDD,键对应的值为可迭代对象时会展开成多个键值对
	>
		x=sc.parallelize([("a", ["x", "y", "z"]), ("b", ["p", "r"])])
		x1=x.flatMapValues(lambda t:t).collect() 
		# x1: [('a', 'x'), ('a', 'y'), ('a', 'z'), ('b', 'p'), ('b', 'r')]
		x2=x.mapValues(lambda t:t).collect()
		# x2: [("a", ["x", "y", "z"]), ("b", ["p", "r"])]
- `.sortByKey(ascending=True,numPartitions=None, keyFunc=<xx>)`:根据键进行排序
- `.sampleByKey(replace, frac, seed=None)`:基于键的采样（分层采样）
- `.substractByKey(other, numPartitions=None)`：基于键的差值，每个(key,val)都位于self而不再other中
- 基于键的聚合操作
	- 常规RDD上fold, aggregate, reduce都是行动操作，pair RDD有对应的针对键进行聚合的转化操作返回新RDD， 新的RDD键为原来的键值是对键的元素聚合的结果
	- `.reduceByKey(f, numPartitions=None, partitionFunc=<>)`: 合并具有相同键的元素， f作用于相同键的那些元素值，不同的键可以并行进行规约操作
	- `.foldByKey(zeroValue, f, numPartitions=none， partitionFunc=<>)` :通过f聚合具有相同键的元素，zeroValue为零值
	- `.aggregateByKey(zeroValue, seqFunc, combFunc, numPartitions=None, partitionFunc=<>)`： 通过f聚合具有相同键的元素
	- `.combineByKey(createCombiner, mergeValue, mergeCombiners, numPartions=None, partitionFunc=<>)`: 最为常用的基于键的聚合函数，大多数聚合函数由它完成.
		- `createCombiner(v)`: v表示键对应的值，返回一个C类型的值(累加器)
		- `mergerValue(c,v)`: c表示当前累加器，v表示键对应的值，返回一个C类型（更新后的累加器)
		- `mergeCombiners(c1,c2)`: c1表示一个分区某个键的累加器，c2表示同一个键另一个分区的累加器，返回合并后的累加器。
		
		基本流程是：遍历分区中元素，考察该元素的键：
		- 如果键从未在该分区中出现过，表明这是分区中的一个新的键。则使用createCombiner() 函数来创建该键对应的累加器的初始值。这一过程发生在每个分区中，第一次出现各个键的时候发生。而不仅仅是整个RDD 中第一次出现一个键时发生
		- 如果键已经在该分区中出现过，则使用mergeValue() 函数将该键的累加器对应的当前值与这个新的值合并
		- 由于每个分区是独立处理的，因此同一个键可以有多个累加器。如果有两个或者更多的分区都有同一个键的累加器，则使用mergeCombiners() 函数将各个分区的结果合并
		>
			x=[('a',1),('a',2),('a',3),('b',4),('b',5),('c',6)] #求各个键的均值，reduceByKey可以快捷实现
			rdd1 = sc.parallelize(x)  # paired RDD
			rdd2 = rdd1.combineByKey(
				    createCombiner=lambda v: (1,v),  
					    # v为当前分区中第一次出现元素的value，返回类型C是(1，v)的元祖，可以理解为该分区第一次出现的元素
					    # 创建了从(k,v)->(k,(1,v))的转化操作，如rdd1中的('a',2)->('a',(1,2))
				    mergeValue= lambda c,v: (c[0]+1, c[1]+v),
					    # 当前分区再次出现相同的键时如处理到’a'对应的('a',3),则将键'a'已经createCombiner出来的C类型(1,2)和
					    # 当前'a'对应的v值3传到mergeValue函数对C进行更新
				    mergeCombiners=lambda c1,c2: (c1[0]+c2[0],c1[1]+c2[1])
					    # 每个partition都处理完成后，将每个分区mergeValue出来的C进行合并，如（'a',1)、('a',2)在partition1, ('a',3)
					    # 在partition2， 则将partiton1返回键'a'的C类型（2,3）和partition2返回键'a'的C类型(1,3)传到mergerCombiners
					    # 进行多分区C类型合并
					)
			z= rdd2.mapValues(lambda v: v[1]/v[0]) #rdd2是(k,C)的格式
			z.persist()
			print(z.collect())
			# z: [('a', 2.0), ('b', 4.5), ('c', 6.0)]

- 基于键的分组操作
	- `.groupByKey(numPartitions=None, partitionFunc=<>)`: 根据键进行分组，返回类型为[K, iterable(V)]的RDD，K为原来RDD的键， V为原来RDD的值
		- 分组是为了聚合，则直接使用聚合操作更好
	- `.cogroup(other, numPartitions=None)`：基于self和other两个RDD中键进行分组，返回类型为[K, (iterable[V],iterable[w])],其中K为两个输入RDD中的键，V为原来self的值， W为other的值
		- 如果一个键只在一个rdd中，则另一个RDD中对应的iterable[]为空
		- `groupWith`的别名，但groupWith支持更多的RDD
- 基于键的链接操作
	- `.join(other, numPartitions=None)`: 两个输入RDD的根据键的内链接，共有的键
	- `.leftOuterJoin(other, numPartitions=None)` ：左外链接，self的键一定存在
	- `.rightOuterJoin(other, numPartitons=None)` ：右外链接，other的键一定存在
	- `.fullOuterJoin(other, numPartitions=None）` : 全连接,合并两者的键

- 两个RDD的操作
	- `substractByKey`
	- `join, rightOuterJoin,leftOuterJoin,cogroup`

### 4.4、行动操作
- 支持所有标准RDD的心动操作，函数参数为二元二组
- `countByKey()`:以字典的形式返回每个键的元素数量
- `collectAsMap()` :以字典的形式返回所有的键值对
- `lookup(key)`:以列表的形式返回指定键的所有的值

### 4.4、数据分区(数据集在节点间的分区)
- 任何需要根据键跨节点进行混洗的操作（如ByKey,Join等)都能从分区获得增益
- `partitionBy(partion obj).persist()`：创建分区表可以减少混洗和网络传输开销
- `HashPartitioner, RangePartitioner`两种内建的分区方式
- 许多spark操作为为结果RDD自动设定分区方式(sortByKey生成范围分区RDD，groupByKey生成hash分区RDD)
- `map`操作会失去基RDD分区信息
- 二元操作的结果分区取决于基RDD的分区方式：
	- 默认情况下采用hash分区
	- 其中一个RDD设置过分区，则结果采用该分区
	- 两种RDD设置过分区，则以第一个输入的分区方式
- **查看分区**
	- `.partioner`:告知rdd中每个键所属的分区
	- `.getNumPartitions` ：查看RDD的分区数
- **指定分区**
	- 聚合分组操作的numPartitions参数指定
	- `.repartition(numPartitions)`:将数据混洗后返回一个拥有指定数量分区数的RDD
	- `.coalesce(numPartitions, shuffle=False)`: 返回一个拥有指定分区数的RDD，新分区数量必须必旧分区少
	- `.partitionBy(numPartitions, partitionFunc)`:返回一个使用指定分区器和分区数量的RDD
		- 新分区可比旧分区大，也可能少
		- partitonFunc是分区函数，如果用多个RDD使用同一分区方式，则应该使用同一个分区函数对象
	- .新分区指定后应该persist，避免重复进行分区操作
- **自定义分区(scala)**
	- 继承org.apache.apark.Partitioner类
	- 实现 numPartition:Int 返回创建的分区数
	- 实现 getPartition(key:Any):Int 返回给定键的分区编号（0 until numPartition-1)
	- 实现equals(): 判断两个分区器是否相等的方法

# 5、spark进阶
### 5.1、spark中的两种共享变量（驱动器和工作节点之间的共享)
- 1.**累加器**:
	- 诞生原因
		- spark在集群中执行代码时会分解每个RDD操作到executor的task中，在此之前会计算每个task的闭包（执行RDD计算时executor需要访问的变量和方法集合），并将闭包的**副本**序列化发送到每个executor。因此在excutor中操作的比如计数器是executor本地的和驱动器中并无直接关系
	- 功能及实现
		- 将工作节点中的值聚合到驱动程序，只支持累加(+=)操作
		- 工作节点对其只写不读，所有工作节点的累计操作最终都自动传播到驱动程序中，在驱动程序中通过.value访问
		- sparkContext累加器只支持基本的数据类型（int, float等）,可通过继承[AccumulatorParam][1]自定义，运算需要满足交换律和结合律
	- 使用流程
		- 驱动程序中创建初始值的累加器： SparkContext.accumulator(init_value)
		- 执行器代码中使用+=或者.add(term)增加值
		- 驱动程序中使用.value访问值
		> 
			file=sc.textFile('xxx.txt')
			acc=sc.accumulator(0)  #初始值为0
			def func(line):
			  global acc #访问全局变量
			  if cond:
			    acc+=1
			  return xx
			rdd=file.map(xxx)

	- 容错功能
		- 行动操作中的累计器，spark确保每个task对累加器修改只应用一次，因此task失败和重新计算时此时的累加器也可靠
		- 转化操作中使用的累加器可能发生不止一次更新，无法保证可靠

- 2.**广播变量**
	- 运行程序高效的向工作节点发送只读值
	- `broacast` 变量的`value`中存放着广播的值，该值只会发送到各节点一次
	- BroadCast的方法和属性：
		- `.destory()`：销毁当前broadcast变量的所有数据和所有metadata，阻塞式方法指导销毁完成
		- `.dump(value,f)`：保持变量
		- `.load(path)`：加载变量
		- `.unpersist(blocking=False)`：删除变量在executor的缓存备份，之后变量被使用需要驱动器程序发送
		- `.value`：返回broadcast变量的值
	- 使用流程
		- 通过`SparkContext.broadcast(xx)`创建
		- 通过`.value `访问值,作为只读值处理
		>
			bd=sc.broadcast(tuple('name','json'))
			def func(row):
			  s=bd.value[0]+row
			  return s
			rdd=rdd.map(func)
	- 广播的优化
	 	- 选择良好的序列化格式，通过`spark.serializer`属性选择序列化库
	 	
# 6、数据读取和存储
支持多种输入输出源,特别是Hadoop MapReduce使用的InputFormat和OutputFormat接口访问数据
，常见数据源：

- 文件格式(文本，json, csv, SequenceFiles, Protobal Buffers)与文件系统(本地，hdfs)
- sparkSQL结构化数据（json, Hive)
- 数据库与键值存储(Hbase, Elasticsearch及JDBC源)

###  6.1、文件格式与文件系统
- 文本文件 
	- val input= sc.testFile(filepath) 支持通配符*
	- val input= sc.wholeTextFile(dir) ->pairRDD[file, context]
	- rdd.saveAsTextFile(path) ->会在路径下输出多个文件
- json
	- 通过jackson包
- csv
	- 通过CSVReader
- sequenceFile (常用Hadoop键值对文件)
	- val data = sc.sequenceFiles(infile, classof[Text],classof[IntWritable])
	- rdd.saveAsSequenceFile(outFile)
- Hadoop格式
	- valinput= sc.newAPIHadoopFile(input, classof[],classof[],classof[],conf)
	- rdd.saveAsNewAPIHadoopFile(file)
- 非文件系统数据源
	- hadoopDataset 访问Hadoop支持的非文件系统，诸如Hbase,MongoDB
- 文件压缩
	- lzo, bzip2支持可分割方式
- 文件系统
	-  本地 :/home/file 指定路径
	-  hdfs : hdfs//master:port/path 指定路径
		
### 6.2、sparkSQL数据
- 连接到hive表: 
     >'import org.apache.spark.sql.hive.HiveContext;'  
	'val hivectx = org.apche.spark.sql.hive.HiveContext(sc);'  
	'val rdd = hivectx.sql('select * from table')';
###  6.3、HBASE
- 通过hadoop输入格式访问Hbase

# 7、SparkSQL基础
### 7.1. 概述
1. spark sql是操作结构化数据的程序包，可以使用SQL或者HQL查询数据，结果以Dataset/DataFrame返回
	- **Dataset**
		- 分布式数据集合提供了RDD和Spark SQL执行引擎的优点
		- 目前Python还不支持，但python的动态特性其很多优点可用
	- **DataFrame**
		- 等价于关系型数据库中的表
		- 在Python中，DataFrame有DataSet中的RowS来表示
2. 支持多种数据源，如hive表，Parquet（默认，列式存储)及Json等
3. 支持SQL和RDD相结合
### 7.2. SparkSession
1. `SparkSession`是spark sql所有功能的入口点。可用于创建`DataFrame`、注册DataFrame为table、在table上执行SQL、缓存table、读写文件等；
2. 创建`SparkSession`
	>
		from pyspark.sql import SparkSession
		spark_session = SparkSession \
		    .builder \     #builder用于创建SparkSession
		    .appName("Python Spark SQL basic example") \ # appName设定web中展示的名字，未指定则随机生成
		    .config("spark.some.config.option", "some-value"，[conf=conf]) \配置，会直接各自传递给SparkContext和SparkSession
			.enableHiveSupport() #开启hive支持
			.master(master=local) # 单机本地运行,master=local[4]本地四核， master=spark://master:7077集群上
		    .getOrCreate() #如果存在一个全局默认的sesseion实例则返回它并用本配置更新，否则根据本配置直接创建

3. SparkSession属性
	- `.builder`
	- `.conf`: 配置接口
	- `.catalog`: 用户通过它操作底层的数据库，表和函数的接口
		>spark_session.catalog.cacheTable('name') #缓存表
	- `.sparkContext`:返回底层的sparkContext
	- 数据读取接口：
		- `.read`： 返回一个DataFrameReader, 从外部存储系统读取数据并返回DataFrame
		-  `.readStream`:返回一个DataStreamReader，将输入数据流读取为DataFrame
		- `.streams`: 返回一个StreamingQueryManager对象，管理当前上下文所有的StreamingQuery
	- `.version`: spark版本

4. SparkSession方法
	- `.createDataFrame(data,schema=None,samplingRatio=None, verifySchema=True)`:从`RDD`、`一个列表`或者`pandas.DataFrame`中创建一个DataFrame
		- schema: DataFrame的结构化信息，可以为：
			- `字符串列表`: 给出列名信息，数据类型从data中推断
			- `None`：要求data为RDD且元素类型为Row,namedtuple,dict之一，以便推断结构化信息
			- `pyspark.sql.types.StructType`:直接指定每列数据的类型
		    - `pyspark.sql.types.DataType`:指定一列数据的类型
		- `samplingRatio`： 指定需要多少比例的纪录来推断数据类型，None只使用第一行
		- `verifySchema`: 根据schema校验每一行数据					
	- `.newSession()`：返回新的Session，但共享`SparkContext`和`table cache`
	- `.range(start, end=None, step=1, numPartitions=None)` 创建只有一列的DataFrame,列名为id,类型为pyspark.sql.types.LongType。
		>
			df = spark_session.range(0,5)
			print(df.collect())
			#[Row(id=0), Row(id=1), Row(id=2), Row(id=3), Row(id=4)]
	- `.sql(sqlQuery)`:sql查询并以DataFrame返回结果
	- `.stop()`: 停止底层的SparkContext
	- `.table(tableName)`: 以DataFrame返回指定的table

### 7.3. DataFrame创建和保存
1.  SparkSession支持从列表，源文件，pandas DataFrame, RDD，HIVE表创建DataFrame
	>
		l = [('Alice', 1)]
		spark_session.createDataFrame(l).collect() #从列表创建
		#[Row(_1=u'Alice,_2=1)]
	>
		sc = spark_session.sparkContext
		rdd = sc.parallelize([('Alice', 1)])
		x = spark_session.createDataFrame(rdd, "name: string, value: int") #从RDD创建
		print(x.collect())
		# x: Row(name=u'Alice',value=1)
	>
		df = pd.DataFrame({'a':[1,3,5],'b':[2,4,6]})
		spark_session.createDataFrame(df).collect()   #从DataFrame创建
		#[Row(a=1, b=2), Row(a=3, b=4), Row(a=5, b=6)]
	
	从`文件`创建,接口是DataFrameReader或者从文件直接加载查询：
	>
		df = spark_session.sql("SELECT * FROM parquet.`examples/src/main/resources/users.parquet`") #文件加载查询
	>	
		reader = spark_session.read
	    #.format(source)通用格式加载
		df = spark_session.read.format('json').load(['python/test_support/sql/people.json','python/test_support/sql/people1.json'])	
	>
		#专用加载
		spark_session.read.json('python/test_support/sql/people.json')
		spark_session.read.text('python/test_support/sql/text-test.txt').collect() #每行文本作一个元素，DataFrame只有一列
		#[Row(value=u'hello'), Row(value=u'this')]
	
	从`HIVE表`创建：
	
	- SparkSQL支持读取和写入Apache Hive中的数据，但Hive具有大量的依赖，如果在类路径找到则spark自动加载，hive依赖也需要存在于所有工作节点
	- 配置： `hive-site.xml`, `core-site.xml`(安全配置）,`hdfs-site.xml`(用户HDFS配置）放在`conf/`目录中
	- SparkSession开启`enableHiveSupport`
		>
			from pyspark.sql import SparkSession
			spark_sess = SparkSession \
			    .builder \
			    .appName("Python Spark SQL Hive integration example") \
			    .config("spark.sql.warehouse.dir", '/home/xxx/yyy/') \
			    .enableHiveSupport() \
			    .getOrCreate()
			spark_sess.sql("CREATE TABLE IF NOT EXISTS src (key INT, value STRING) USING hive")
			spark_sess.sql("LOAD DATA LOCAL INPATH 'examples/src/main/resources/kv1.txt' INTO TABLE src")
			spark.sql("SELECT * FROM src").show()
	- 创建Hive 表时，需要定义如何向/从文件系统读写数据(输入输出格式），还需要定义该表的数据的序列化与反序列化。可以通过在OPTIONS 选项中指定这些属性：
		>
			spark_sess.sql("CREATE TABLE IF NOT EXISTS src (key INT, value STRING) USING hive OPTIONS(fileFormat 'parquet')")
	
		可用的options有：
		- `fileFormat`：文件格式，支持`sequencefile`、`rcfile`、`orc`、`parquet`、`textfile`、`avro`
		- `inputFormat,outputFormat`:成对出现，如果指定了fileFormat则无需指定；inputFormat 'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat'
2. DataFrame的保存通过`DataFrameWriter`完成
	- 通过`DataFrame.write`来访问`DataFrameWriter`
	- 通用保存：
		- `.format(source)`:设置数据格式
		- `.mode(saveMode)`:设置保存模式,`append,overwrite,ignore,error`
		- `.paritionBy(*cols)：按指定的列名将输出的DataFrame分区
		- `.save(path=None, format=None,mode=None,partitionBy=None,**options):保存DataFrame
		>
			df.write.format('json').mode('append').partitionBy('colname').save('./data.json')
	- 专用保存
		- `.csv()`
		- `.insertInto(tableName,overwirte=False)`
		- `.json()`
		- `.saveAsTable()`
		- `.text()`
		- 等
### 7.4. [DataFrame][2]的属性和方法
1.  **概述**
	- 一个`DataFrame` 实例代表了基于命名列的分布式数据集
	- 通过df.key和df[key] 访问`DataFrame`的列,并不支持pandas DataFrame其他的切片索引方式
2. **属性**
	- `.columns`:列表的形式返回所有的列名
	- `.dtypes`: 以列表的形式返回所有列的名字和类型[(col1_name, col1_type),(col2_name,col2_type)]
	- `.isStreaming`: 是否包含数据流
	- `.na` : 返回一个DataFrameNaFunctions对象，处理缺失值
	- `.rdd`: 返回DataFrame的底层RDD（元素类型为Row）
	- `.schema`: 
	- `.stat`
	- `.storageLevel`
	- `.write`: 返回一个DataFrameWrite对象，`no-streaming DataFrame`的外部存储接口
	- `.writeStream`:返回一个DataStreamWriter对象，它是`streaming DataFrame`的外部存储接口
	
3. **方法**

	基本囊括了Hive和pandasDataFrame的基本方法。

	1. **转化方法**
		1. 聚合操作
			- `.agg(*exprs)`:在整个DataFrame开展聚合操作(`df.groupBy.agg()`的快捷方式）		
				>
					df.agg({"age": "max"}).collect() #在 agg 列上聚合
					# 结果为：[Row(max(age)=5)]
					# 另一种方式：
					from pyspark.sql import functions as F
					df.agg(F.max(df.age)).collect()
			- `.filter(cond)`:对行进行过滤
				- `.where（）`的别名
				- `cond`: 一个types.BooleanType的Column,或一个字符串形式的SQL表达式
				>
					df.filter(df.age > 3).collect()  
					df.filter("age > 3").collect()
					df.where("age = 2").collect()
		- 分组操作（返回`GroupedData`对象)
			- `.cube(*cols)`:根据当前DataFrame 的指定列，创建一个多维的cube，从而方便我们之后的聚合过程
			- `.groupBy(*cols)`: 通过指定的列来将DataFrame 分组，从而方便我们之后的聚合过程。
			- `.rollup(*cols)`:创建一个多维的rollup，从而方便我们之后的聚合过程。
			
		- 排序操作（返回一个新的`DataFrame`)
			- `.orderBy(*cols, **kw)`:它根据旧的DataFrame 指定列排序,返回一个新的DataFrame
				- `cols`：一个列名或者Column 的列表，指定了排序列
				- `ascending`:一个布尔值，或者一个布尔值列表(和cols长度相同)
			- `.sort(*cols, **kw)`:它根据旧的DataFrame 指定列排序,返回一个新的DataFrame
				- `cols`：一个列名或者Column 的列表，指定了排序列
				- `ascending`:一个布尔值，或者一个布尔值列表(和cols长度相同)
			>
				from pyspark.sql.functions import *
				df.sort(df.age.desc())
				df.sort("age", ascending=False)
				df.sort(asc("age"))
				​
				df.orderBy(df.age.desc())
				df.orderBy("age", ascending=False)
				df.orderBy(asc("age"))
			- `.sortWithinPartitions(*cols, *kw)`:返回一个新的DataFrame，它根据旧的DataFrame 指定列在每个分区进行排序
		- 调整分区
			- `.coalesce(numPartitions)`:返回拥有指定的numPartitions 分区的新DataFrame
				- 只能缩小分区数量，而无法扩张分区数量。如果numPartitions 比当前的分区数量大，则新的DataFrame 的分区数与旧DataFrame 相同
				- 不会混洗数据
			- `.repartition(numPartitions, *cols)`:返回一个新的DataFrame，拥有指定的numPartitions 分区
				- 结果DataFrame 是通过hash 来分区
				- 它可以增加分区数量，也可以缩小分区数量
		- 集合操作
			- `.crossJoin(other)`:返回一个新的DataFrame，它是输入的两个DataFrame 的笛卡儿积
				>
					可以理解为 [row1,row2]，其中 row1 来自于第一个DataFrame，row2 来自于第二个DataFrame
			- `.itersect(other)`:返回两个DataFrame 的行的交集
			- `.join(other,on=None,how=None)`:返回两个DataFrame的 join
			- `.substract(other)`:返回一个新的DataFrame，它的行由位于self 中、但是不在other 中的Row 组成
			- `.union(other)`:  返回两个DataFrame的行的并集（它并不会去重）,`unionAll`的别名
		- 统计
			- `.crosstab(col1,col2)`:统计两列的成对频率。要求每一列的distinct 值数量少于 `1e4` 个。最多返回 `1e6`对频率
				- 是`DataFrameStatFunctions.crosstab()`的别名
				- 结果的第一列的列名为，col1_col2，值就是第一列的元素值。后面的列的列名就是第二列元素值，值就是对应的频率
				>
					df =pd.DataFrame({'a':[1,3,5],'b':[2,4,6]})
					s_df = spark_session.createDataFrame(df)
					s_df.crosstab('a','b').collect()
					#结果： [Row(a_b='5', 2=0, 4=0, 6=1), Row(a_b='1', 2=1, 4=0, 6=0), Row(a_b='3', 2=0, 4=1, 6=0)]
			- `.describe(*cols)`: `pandas.DataFrame.describe()`
			- `.freqItem(cols, support=None)`:寻找指定列中频繁出现的值（可能有误报）
				- 是`DataFrameStatFunctions.freqItems()` 的别名
				- `support`： 指定所谓的频繁的标准（默认是 1%）。该数值必须大于  
		- 移除数据
			- `.distinct()`:返回一个新的DataFrame，它保留了旧DataFrame 中的distinct 行
			- `.drop(*cols)`:返回一个新的DataFrame，它剔除了旧DataFrame 中的指定列,cols不存在时不做任何操作
			- `.dropDuplicates(subset=None)/.drop_duplicates`:返回一个新的DataFrame，它剔除了旧DataFrame 中的重复行
				- 它与`.distinct()` 区别在于：它仅仅考虑指定的列来判断是否重复行
				- `subset`:列名集合（或者Column的集合）。如果为None，则考虑所有的列
			- `.dropna(how='any', thresh=None, subset=None)`:返回一个新的DataFrame，它剔除了旧DataFrame 中的null行
				- 它是`DataFrameNaFunctions.drop()` 的别名
				- `how`：指定如何判断null 行的标准。'all'：所有字段都是na，则是空行；'any'：任何字段存在na，则是空行
				- `thresh`：一个整数。当一行中，非null 的字段数量小于thresh 时，认为是空行。如果该参数设置，则不考虑how
				- `subset`：列名集合，给出了要考察的列。如果为None，则考察所有列
			- `.limit(run)`: 返回一个新的DataFrame，它只有旧DataFrame 中的num行
		- 采样、拆分
			- `.randomSplit(weights, seed=None)`：返回**一组**新的DataFrame，它是旧DataFrame 的随机拆分
			- `.sample(withReplacement, fraction, seed=None)`：返回一个新的DataFrame，它是旧DataFrame 的采样
			- `.sampleBy(col, fractions, seed=None)`：返回一个新的DataFrame，它是旧DataFrame 的采样,它执行的是无放回的分层采样,分层由col 列指定
				- `col`：列名或者Column，它给出了分层的依据
				- `fractions`：一个字典，给出了每个值分层抽样的比例。如果某层未指定，则其比例视作 0
				>
					sampled = df.sampleBy("key", fractions={0: 0.1, 1: 0.2}, seed=0)
					# df['key'] 这一列作为分层依据，0 抽取 10%， 1 抽取 20%
		- 替换
			- `.replace(to_replace, value=None, subset=None)`：返回一组新的DataFrame，它是旧DataFrame 的数值替代结果
				- 是`DataFrameNaFunctions.replace()` 的别名
				- `to_replace`：可以为布尔、整数、浮点数、字符串、列表、字典，给出了被替代的值;如果是字典，则给出了每一列要被替代的值
				- `value`：一个整数、浮点数、字符串、列表。给出了替代值
				- `subset`：列名的列表。指定要执行替代的列
			- `.fillna(value, subset=None)`：返回一个新的DataFrame，它替换了旧DataFrame 中的null值
				- 它是`DataFrameNaFunctions.fill()`的别名
				- `value`：一个整数、浮点数、字符串、或者字典，用于替换null 值。如果是个字典，则忽略subset，字典的键就是列名，指定了该列的null值被替换的值
				- `subset`：列名集合，给出了要被替换的列
		- 选数
			- `.select(*cols)`: 执行一个表达式，将其结果返回为一个DataFrame
				- `cols`：一个列名的列表，或者Column 表达式。如果列名为*，则扩张到所有的列名
				>
					df.select('*')
					df.select('name', 'age')
					df.select(df.name, (df.age + 10).alias('age'))
			- `.selectExpr(*expr)`: 执行一个SQL 表达式，将其结果返回为一个DataFrame
				- `expr`：一组SQL 的字符串描述
				> df.selectExpr("age * 2", "abs(age)")
			- `.toDF(*cols)`: 选取指定的列组成一个新的DataFrame
				- `cols`：列名字符串的列表
			- `.toJSON(use_unicode=True)`: 返回一个新的DataFrame，它将旧的DataFrame 转换为RDD（元素为字符串），其中每一行转换为json 字符串。
		- 列操作
			- `.withColumn(colName, col)`：返回一个新的DataFrame，它将旧的DataFrame 增加一列（或者替换现有的列）
				- `colName`：一个列名，表示新增的列（如果是已有的列名，则是替换的列）
				- `col`：一个Column 表达式，表示新的列
				>df.withColumn('age2', df.age + 2)
			- `.withColumnRenamed(existing, new)`：返回一个新的DataFrame，它将旧的DataFrame 的列重命名
				- `existing`:一个字符串，表示现有的列的列名
				- `col`：一个字符串，表示新的列名
	2. **行动操作**
		1. 查看数据
			- `.collect()`: 以Row 的列表的形式返回所有的数据
			- `.first()`: 返回第一行（一个Row对象）
			- `.head(n=None)`:返回前面的n 行
				- 如果返回1行，则是一个Row 对象
				- 如果返回多行，则是一个Row 的列表
			- `.show(n=20,truncate=True)`: 在终端中打印前 n 行,它并不返回结果，而是print 结果
			- `.take(num)`:以Row 的列表的形式返回开始的num 行数据
			- `.toLocalIterator()`：返回一个迭代器，对它迭代的结果就是DataFrame的每一行数据（Row 对象）
		2. 统计
			- `.corr(col1, col2, method=None)`：计算两列的相关系数，返回一个浮点数。当前仅支持皮尔逊相关系数
				- `DataFrameStatFunctions.corr()`的别名
			- `.cov(col1,col2)`：计算两列的协方差。
				- `DataFrameStatFunctions.cov()`的别名
			- `.count()`：返回当前DataFrame 有多少行
		3. 遍历
			- `.foreach(f)`：对DataFrame 中的每一行应用f，它是`df.rdd.foreach()` 的快捷方式
			- `.foreachPartition(f)`：对DataFrame 的每个分区应用f，它是`df.rdd.foreachPartition()` 的快捷方式
			>
				def f(person):
				    print(person.name)
				df.foreach(f)
				​
				def f(people):
				    for person in people:
				        print(person.name)
				df.foreachPartition(f)
			- `.toPandas()`：将DataFrame 作为`pandas.DataFrame` 返回
	3. **其他方法**
		- 缓存
			- `.cache()`：使用默认的storage level 缓存（缓存级别为：MEMORY_AND_DISK ）
			- `.persist(storageLevel=StorageLevel(True, True, False, False, 1))`： 缓存DataFrame
			- `.unpersist(blocking=False)`：标记该DataFrame 为未缓存的，并且从内存和磁盘冲移除它的缓存块。
		- `.isLocal()`：如果`collect()` 和 `take()` 方法能本地运行（不需要任何executor 节点），则返回True。否则返回False
		- `.printSchema()`：打印DataFrame 的 schema
		- `.createTempView(name)`：创建一个临时视图，name 为视图名字
			- 临时视图是session 级别的，会随着`session` 的消失而消失
			- 如果指定的全局临时视图已存在，则抛出`TempTableAlreadyExistsException`异常
			>
				df.createTempView("people")
				df2 = spark_session.sql("select * from people")	
		- `.createOrReplaceTempView(name)`：创建一个临时视图，name 为视图名字。如果该视图已存在，则替换它。
		- `.createGlobalTempView(name)`：创建一个全局临时视图，name 为视图名字
			- spark sql 中的临时视图是`session` 级别的，会随着session 的消失而消失。如果希望一个临时视图跨session 而存在，则可以建立一个全局临时视图
			- 全局临时视图存在于系统数据库`global_temp` 中，必须加上库名取引用它
			>
				df.createGlobalTempView("people")
				spark_session.sql("SELECT * FROM global_temp.people").show()

		- `.createOrReplaceGlobalTempView(name)`：创建一个全局临时视图，name 为视图名字。如果该视图已存在，则替换它
		- `.explain(extended=False)`：打印`logical plan` 和`physical plan`，用于调试模式
			- `extended`：如果为False，则仅仅打印`physical plan`



### 7.5. [Row][2]
1. 一个`Row` 对象代表了`DataFrame` 的一行
2. 可以通过两种方式来访问一个`Row` 对象：
	- 通过属性的方式：`row.key`
	- 通过字典的方式：`row[key]`
3. `key in row` 将在`Row` 的键上遍历（而不是值上遍历）
4. 创建`Row`：通过关键字参数来创建
	>row = Row(name="Alice", age=11)
5. 可以创建一个`Row` 作为一个类来使用，它的作用随后用于创建具体的`Row`
	>
		Person = Row("name", "age")
		p1 = Person("Alice", 11)
6. `.asDict(recursive=False)`：以字典的方式返回该`Row` 实例。如果`recursive=True`，则递归的处理元素中包含的`Row`

### 7.6. [Column][2]
- `Column` 代表了`DataFrame` 的一列
- 有两种创建`Column` 的方式：
	- 通过`DataFrame` 的列名来创建：
	>
		df.colName
		df['colName']
	- 通过`Column` 表达式来创建：
	>
		df.colName+1
		1/df['colName']
1. 方法
	- `.alias(*alias, **kwargs)`：创建一个新列，它给旧列一个新的名字（或者一组名字，如explode 表达式会返回多列）
		- 它是`name()`的别名
		- `metadata`：一个字符串，存储在列的`metadata` 属性中
		>
			df.select(df.age.alias("age2"))
			# 结果为： [Row(age2=2), Row(age2=5)]
			df.select(df.age.alias("age3",metadata={'max': 99})).schema['age3'].metadata['max']
			# 结果为： 99
	- 排序
		- `.asc()`:创建一个新列，它是旧列的升序排序的结果
		- `.desc()`：创建一个新列，它是旧列的降序排序的结果
	- `.astype(dtype)`:创建一个新列，它是旧列的数值转换的结果(`.cast()`的别名）
	- `.between(lowerBound, upperBound)`：创建一个新列，它是一个布尔值。如果旧列的数值在[lowerBound, upperBound]（闭区间）之内，则为True
	- 逻辑运算
		- `.bitwiseAND(other)`：二进制逻辑与
		- `.bitwiseOR(other)`：二进制逻辑或
		- `.bitwiseXOR(other)`：二进制逻辑异或
	- 元素抽取
		- `.getField(name)`:返回一个新列，是旧列的指定字段组成。此时要求旧列的数据是一个`StructField（如Row）`
			>
				df = sc.parallelize([Row(r=Row(a=1, b="b"))]).toDF()
				df.select(df.r.getField("b"))
				#或者
				df.select(df.r.a)
		- `.getItem(key)`：返回一个新列，是旧列的指定位置（列表），或者指定键（字典）组成
			- key：一个整数或者一个字符串
			>
				df = sc.parallelize([([1, 2], {"key": "value"})]).toDF(["l", "d"])
				df.select(df.l.getItem(0), df.d.getItem("key"))
				#或者
				df.select(df.l[0], df.d["key"])
	- 判断
		- `.isNotNull()`:返回一个新列，是布尔值。表示旧列的值是否非null
		- `.isNull()`：返回一个新列，是布尔值。表示旧列的值是否null
		- `.isin(*cols)`：返回一个新列，是布尔值。表示旧列的值是否在cols 中
			>
				df[df.name.isin("Bob", "Mike")]
				df[df.age.isin([1, 2, 3])]
		- `.like(other)`：返回一个新列，是布尔值。表示旧列的值是否like other。它执行的是SQL 的 like 语义
			>df.filter(df.name.like('Al%'))
		- `.rlike(other)`：返回一个新列，是布尔值。表示旧列的值是否rrlike other。它执行的是SQL 的 rlike 语义
	- 字符串操作（`other` 为一个字符串）
		- `.contains(other)`：返回一个新列，是布尔值。表示是否包含other
		- `.endswith(other)`：返回一个新列，是布尔值。表示是否以other 结尾
			>df.filter(df.name.endswith('ice'))
		- `.startswith(other)`：返回一个新列，是布尔值。表示是否以other 开头
		- `.substr(startPos, length)`：返回一个新列，它是旧列的子串
	- `.when(condition, value)`：返回一个新列
		- 对条件进行求值，如果满足条件则返回value，如果不满足：
			- 如果有`.otherwise()` 调用，则返回`otherwise` 的结果
			- 如果没有`.otherwise()` 调用，则返回`None`
			- `condition`：一个布尔型的Column 表达式
			- `value`：一个字面量值，或者一个Column 表达式
			>
				from pyspark.sql import functions as F
				df.select(df.name, F.when(df.age > 4, 1).when(df.age < 3, -1).otherwise(0))
			- `.otherwise(value)`：value 为一个字面量值，或者一个Column 表达式
	

### 7.7. [GroupedData][2]
通常由`DataFrame.groupBy()` 创建，用于分组聚合

1. **方法**
	- `.agg(*expr)`:聚合并以DataFrame 的形式返回聚合的结果
		- 可用的聚合函数包括：`avg、max、min、sum、count`
		- `exprs`：一个字典，键为列名，值为聚合函数字符串。也可以是一个Column 的列表
		>
			df.groupBy(df.name).agg({"*": "count"}) #字典
			# 或者
			from pyspark.sql import functions as F
			df.groupBy(df.name).agg(F.min(df.age)) #字典
	- .直接调用聚合函数,
		- `.avg(*cols)`
		- `.count()`
		- `.max(*cols)`
		- `.min(*cols)`
		- `.sum(*cols)`
	- `.pivot(pivot_col, vlaues=None)`:对指定列进行透视
		- `pivot_col`：待分析的列的列名
		- `values`：待分析的列上，待考察的值的列表。如果为空，则spark 会首先计算`pivot_col` 的 `distinct` 值
		>
			df4.groupBy("year").pivot("course", ["dotNET", "Java"]).sum("earnings")
			#结果为：[Row(year=2012, dotNET=15000, Java=20000), Row(year=2013, dotNET=48000, Java=30000)]
			# "dotNET", "Java" 是 course 字段的值

### 7.8. 内建的[functions][2]
pyspark.sql.functions，包含了常见是数学计算和字符串函数
# 8、快捷浏览
1. **基于分区进行操作**
	- mapPartitions() ->调用分区中元素的迭代器，返回分区中元素的迭代器
	- mapPartitionsWithIndex()
	- foreachPartition() ->调用元素迭代器， 无返回

2. **外部程序通讯的管道**
	- pipe() 可让任何一种能读写unix标准流的语言实现spark中的作业逻辑
	- sc.addFile(path)->构建文件列表，让每个工作节点在spark作业中下载列表中的文件

3. **数值RDD操作**
	- 调度stats()一次性得出或分别各自调用得出
	- `count()， mean(),  sum(), min(),max(), variance(), sampleVariance(), stdev(), sampleStdev()`

[GitHub](https://github.com/sameul-yuan 'myself')
   
[1]:http://spark.apcahe.org/docs/latest/api/scala/index.html#packages "包文档"
[2]:http://huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/spark/chapters/03_dataframe.html
