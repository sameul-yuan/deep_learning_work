# scala基本知识
## 1. scala的语法:
### 1.0 六大特性
1. 和java无缝集成，运行在JVM上，编译成.class字节码
2. 自动类型推断，val，var就可以
3. 支持分布式和并发Actor（java中的threading）
4. 特质： java中的抽象类和接口功能的集合
5. 模式匹配：switch语句中不仅可以匹配基础对象，还可以直接匹配类型
6. 函数式语言，函数能当值使用；
### 1.1 基本数据类型
- val a =1 值类型，定义后无法改变，推荐使用便于垃圾回收（不可变类型，如果后面没用会回收内存）
- var a=1  可变变量，
- val a:Int = 1 变量类型为Int，变量和函数的返回类型写在后面 val name:type
- Byte,Char,Short, Int, Long,Float, Double, Boolean, BigInt, BigDecimal,String
- Null(具有唯一实例null)，None（Option的子类，另一个是Some)，Any(Scala中基础基类），AnyRef，Unit(其他语言的void, 没有返回结果的类型), Nothing(推断不出的类型),Nil(0个元素的List)
- +-*/%  to/until等操作符实际上是方法; val a = 1 to 10 , val a = 1.to(10)
- val result = for( i<- 1 to 1000 if(i<500) if (i%2==0)) yield i  推导
- 字符串插值 s"this is ${}"  ， 类似python的f-string

### 1.1 控制结构
- scala每行语句的后面会自动补全分号；

- if/else 结构的值是之后表达式的值： 
  
    >var r = if(x>0) 0 else 1
    
- {} 块表达式最后一个表达式的值就是块返回值,最后一个表达式是语句并没有返回值则返回Unit, 如果函数体中if语句返回需要显示的return 语句；

    > val c = { val a=2;  val b=2; a*b}   // c=4
    >
    > val c= { val a=2
    >
    > ​             val b=2
    >
    > ​              if (a>2 && b==2) { a+1} else {a}
    >
    > ​            }

- while(n>1) ...; 

- do .... while ...

- for 
    ```scala
    //推导：将生成器、条件过滤和定义组合在一起 ;() 或{}
    for(i<- 1 to n)  包含n
    for(i<- 1.to(10))
    for(i <- 1 until n) 不包含n
    for(i <- Range(0,n)) 不包含n for(i<-Range(0,n).inclusive)
    for(i<- 1 to n; j<- 1 to n) 用;分隔多个生成器
    for(i<- 1 to n if i%2==0; j<- 1 to n if j%2==0 if j>3) 每个生成器可以带一个或多个if的守卫
    
    val vec = Vector(1,2,3,4) //Vector[int](1,2,3,4)
    val result = for{i<-vec if i>2} yield i //result的类型会自动从vec中推断出来
    val result = for(i<-vec if i%2==1) yield {val v=i+1; v+1} //yield可以与表达式结合并且 yield可以嵌套
    
    ```
    
### 1.2 类和对象
- 当new类时，类中除了方法不执行(构造方法执行)其他都执行

- Scala程序必须从一个对象的main方法开始
    ```scala 
    object Main {
    def main(args:Array[String]):Unit={}
    }
    #或者
    object Main extends app{
        println()
    }
    ```
    
- object相当于java中的单例，定义的都的静态的，没有括号传参,可以实现apply方法实现调用时传参,高效地共享单个不可变实例;   Array，List, Vector等都是object；  不能new, 定义类时同时创建了唯一的对象，就是本身；
    ```scala
    object Lesson{ 
        def main(args:Array[String]):Unit={
            val p = New Person('zhangshan',20,'M') #scala中new class时，类中除了方法不执行，其他都执行
            Lession(10) #调用apply
        }
    
        def apply(x:Int):Unit={  #除非定义了apply方法（传参时执行apply方法，也称工厂方法,可以创建其他对象）
            println("apply called")
        }
    }
    ```
    
- class 可以传参（scala特有）,传参时需要指定类型便于类中自动类型推断，有了参数就有默认的构造，类中属性默认有getter和setter方法，对于属性age, getter和setter方法分别为 def age(){} def age_={}
    ```scala
    class Person( xname:String,xage:Int){ # xname,xage只有Person可见，如果是Person(val/var xname:String ,xage:Int) 则xname在类外可访问
        val name = xname        #默认public访问级别
        private val age = xage  #设为private访问级别
        var gender = 'F'
        def  sayName():Unit={  #Unit表示为返回void
            println("hello world")
        }
        def this(xname:String, xage:Int, xgender:Char){
            this(xname, xage)    # 重写构造时，第一行必须先调用默认(主)构造器(按照类参数列表和类体产生)或其他辅助构造器
            this.gender = gender 
        }
    }
    ```
    
- 如果是object Person,则class Person类是该object Person对象的伴生类，该Person对象是Person类的伴生对象，可以互相访问私有变量；而且伴生类的多个对象共享伴生对象创建的变量

    ```scala
    class Person{  //伴生类
        def increment()={Person.X += 1; Person.X}  //Person.X 对伴生对象的访问
    }
    object Person{ //伴生对象
        var X:Int = 0  //共享该值
    }
    
    p1 = new Person()
    p2 = new Person()
    p1.increment() // 1
    p2.increment() // 2
    ```

    

- Trait,相当于Java中接口和抽象类结合
    - 可以在Trait中定义方法实现或不实现，var/val都可以，未实现的方法就想抽象的
    - 不可以传参（抽象类可以传参）
    - 类继承Trait 第一个关键字是extends，之后用with
        ```scala
        class A extends trait1 with traint2{} # with实现多个特质，后面的特质先执行
        ```
    - 继承类中需要实现没实现的方法，否则强制为abstract 类

- 类继承
    ```scala
    class A extends B{}
    override def toString = super.toString  #用override重写超类的非抽象方法,super调用超类方法
    abstract class  A #定义抽象类，不能实例化
    def sum:Int  #抽象方式省去了方法体，子类中重写抽象字段、方法时不需要override
    val att:Int # 抽象字段没有初始值的字段
    
    class A(arg1, arg2,arg3) extends B(arg1,arg2){}//基类有构造器参数则必须显示提供,arg1,arg2为传给基类构造器的参数
    // A的重载构造器必须调用其他构造器，不能调用基类构造器， 只有A的主构造器是调用基类构造器的唯一门户（自动实现)
    ```


### 1.3 方法和函数
1. 方法的定义
    - def func(arg1: type1, arg2:type2):returnType={ line  of code ; return_val} //方法体是表达式
    
    - 方法中传参需要指定类型
    
    - 如果没写return返回值，方法可以自动推断返回类型，模型将最后一行计算的结果当做返回值
    
    - 如果要写return返回值，需显示的声明返回类型
    
    - 如果方法体可以一行搞定，方法体的{}可省略（方法体是表达式)
        ```scala
        def abs(x:Double=10):Int = if (x>=0) x else -x  
        ```
        
    - 如果方法名和方法体之间没有= 则为过程，返回Unit 
    
        ```scala
        def print(args:Int*) {for(i<- args) println(i)}
        ```
    
    - 不带参数的Scala方法通常不使用圆括号，一般没有参数且不改变当前对象的方法不带圆括号
    
        ```scala
        def fun = 3  # 不带参数(调用时不需要括号); 自动推断返回类型为Int;只有一行方法体不需要{} 的方法； 
        def fun {println(3)} # 不带参数的过程,方法体需要加{}
        ```
    
2. 递归方法
   
    - 显示声明返回类型
    
3. 参数可以有默认值
    ```scala
    def fun(x:Int=100):Unit={}
    ```
    
4. 可变长参数
    ```scala
    def sum(args: Int*)={ for (arg<-args) result+=arg} 
    def sum(args: Int*)={args.foreach(ele=>{println(ele)}} 
    ```
    
4. 参数序列（Python中的*)
    ```scala
    sum(1 to 5:_*) #使用_*将区间转成参数序列
    ```
    
6. 匿名函数
    - `()=>{}`

        ```scala
        n=>3 //只有一个参数且可推断出返回类型可去掉(), 表达式只有一行可去掉{}
        ```

    - 可以赋值给一个变量，供调用
        ```scala
        //函数表里的声明 funame: (arg1Type,arg2Type,..)=>returnType
        var fun：String=>Unit = (a:String)=>{println(a)} // 只有一个参数声明的()可去掉
        def fun: (int, int)=>int=(a:Int,b:Int)=>{a+b}
        ```
        
    - 常见用法
        ```scala
        (x:Double)=> 3*x  #匿名函数
        val f = (x:Double)=>3*x  #匿名函数赋给变量 def f = {(x:Double)=>3*x}
        Array(1,2,3).map((x:Double)=>3*x)
        Array(1,2,3).map((x)=>3*x) #由于map知道传入的类型，Double可去掉
        Array(1,2,3).map(x=>3*x) #只有一个参数x，可省略()
        Array(1,2,3).map(3*_) #参数在=>右侧只出现一次可用_代替
        ```
    
7. 偏应用函数
    ```scala
    def fun: String=>Unit=showLog(2020,_:String) # fun('a') = showLog(2020,'a')
    ```

8. 偏函数

  - def fun:PartionlFunction[String,int]={case ...}

   - PartionlFunction[匹配类型，返回类型]

   - java中的switch case

   - 不能带括号传参，只能匹配一个值返回一个值，如下匹配string返回int

     ```scala
     def MyTest:PartionlFunction[String,Int]={
     case "abc"=>2
     case "a"=>1
     case _=>100
     }
     ```

8. 嵌套方法

9. 高阶函数
    - 参数是函数
      
        > def fun(f:(int,int)=>Int, s:String):Unit={...}
    - 返回类型是函数，显示的指示函数的返回是函数(在返回的函数后面加 _ 可以不用指明）
        > def fun(s1:String,s2:String):(string,string)=>String={...}
        > val f = fun _  #将函数赋给变量，_表示方法代表的这个函数
    - 参数和返回都是函数 

10. 柯里化函数
    - 高阶函数的简化（返回值类型是函数)
    ```scala
    def fun(a:Int,b:Int)(c:Int,d:Int):Int={
        a+b+c+d
    }
    ```

### 1.4 String 
- 就是Java中的String 
- 不可变，不可new
- println(s"this is $name") ；python中f-string功能类似
### 1.5 集合
1. Array
    - Array固定长度数据
    - 不可变，可new
    ```scala
    val nums = new Array[Int](10)  #10个初始值为0的整形数组
    val strs = Array('hello','world')
    array(0)  = Array[int](1,2,3)
    array(1) = Array[int](3,4,5)
    for(ele<-array; mem<-ele)
    {
        println(mem)
    }
    array.foreach(ele=>{ele.foreach(println)})
    Array.concat(arry1,arry2)
    Array.fill(5)('string')

    val array = new Array[Array[int]](3) #二维数组

    ```
    
2. ArrayBuffer 
    - import scala.collection.mutalbe.ArrayBuffer
    - ArrayBuffer长度可变数组,提供初始值不用new
    ```scala
    val b =ArrayBuffer[Int]() #初始化一个空的可变长数组
    b += 1
    b += (1,2,3) #最加元素
    b ++= Array(1,2,3) #追加集合
    b.trimEnd(2) #移除最后2个元素
    b.insert(pos, num1,num2)
    b.toArray  #转成定长
    b.boBuffer #转成可变长度
    ```
3. List 
    - ()访问元素
    - `val list = List[String](...)`
    - map
    - flatMap
    ```scala
    val list=List[Int](1,2,3)
    val list.count(ele=>{})
    val list.filter(ele=>{})
    val result = list.map(s=>{s+1})
    val result2 = list.flatMap( ) 
    for(elem <- arr)  #来遍历元素
    for(elem <- arr if ..) yield .. #将原数组转变为新数组；  
    ```
4. ListBuffer 
    ```scala
    val list=ListBuffer[Int](1,2,3)
    ```
5. Set 
    - mutable.Set ,imutableSet
    ```scala
    val set = Set[Int](1,2,3,1) #没有重复元素
    val set2 = Set[Int](2,3,4)
    val result = set&set2  #(&交集，&~差集)

    import scala.collections.mutable.Set #自动找最近定义的
    val set = Set[Int](1,2,3)  # val set = mutable.Set[Int](1,2,3)
    set.+=(100)
    ```
6. Map
    - map中的元素就是一个个的(k,v) 元祖
    - mutable.Map, imutable.Map 
    - HashMap
    ```scala
    val score= Map[String,Int]('al'->90,'bl'->90,('c',90)) #默认是不可变的
    val score = scala.collection.mutable.Map('al'->90)
    score.put('a3',100)
    val score = new scala.collection.mutable.HashMap[String,Int] #构造空映射，需要选定映射实现
    score(key)  #取值
    score.get(key).get  #如果key存在通过get获取返回的Some内容
    score.get(key).getOrElse(100)
    score.getOrElse('al',defualt) 
    score.contains(key)
    score +=('sl'->90,'tl'->90) #添加映射
    score -='sl'  #移除键值对
    for((k,v)<- score) #遍历
    map1.++:(map2) # map1更新map2
    map1.++(map2) # map2更新map1
    ```
    -  map遍历时(k,v)组成元祖 tp._1,tp_2
7. Tuple
    - 最多22个元素
    - val iter = tuple.productIterator
    - 二元祖中有swap翻转函数
    - 对象(不一定相同类型)的集合，可new可不new，可直接写元素
    ```scala
    val (a,b,c)=(1,1,'a') 模型匹配获取元祖的值,
    val res =(1,2,3)
    res._1 ,res._2对应元祖中第一个和第二个元素

    val tp1 = new Tuple1("hello")
    val tp2 = new Tuple2("hello","that")

    val iter:Iterator[Any] = tuple2.productIterator #迭代
    iter.foreach(println)

    tp1.toString
    ```
8. 常见的集合操作
    - 所有的集合都是iterable的
    - 集合包含seq，set、map三大类  
        1. 序列：如数组、列表可通过下标索引
        2. set：如Set无须的， LinkedHashSet 保留插入顺序，SortedSet排序的
        3. map: 如Map， SortedMap按键排序
        >+将元素添加到无序的Set
            +：和:+分别将元素前向和后向添加到序列
            ++将两个机会拼接
            -和--删除元素
        
- 列表：要么是Nil(空列表),要么是一个head元素+tail列表
        ::从给定的头尾创建列表， ::是右结合类别将从末端开始构建
    
- s.map(fun) 
    
    - s.flatMap(fun) 展平映射  (spark中rdd调用的map，flatMap就是scala中的实现)
    
- s.reduceLeft/Right(op)
    - s.foldLeft(init)(op) 提供了额外初始值的reduce
    
    - s.scanLeft/Right(op) 产生中间结果和最后的结果
    
    - print(List(1,2,4).reduceLeft(_ + _) #每个元素只出现一次时用`_`代替
    
    - a zip b  //a.zip(b)
    
    - a.zipAll(b,default_a, defualt_b)
    
    - a.zipWithIndex
    
      

### 1.6 模式匹配
- match ... case ...
- 既可以匹配值，也可以匹配类型
- 从上往下匹配，匹配上自动break
- 匹配过程中会有值的转换(如1.0 转成int 1)
- _ 最后的默认匹配，default 
- 模式匹配语句相当于一大行
1. 普通值匹配
    ```scala
    val ch ='+'
    val sign = ch match{
    case ‘+’ =>1
    case '-' => -1
    case parm if Character.isDigit(parm) => Character.digit(ch,10)  #case后接变量则ch会赋给变量，用if给模式添加守卫
    case _ =>0      #_代表default
    }
    ```

2. 类型匹配
    ```scala
    val sign = obj match{
    case x:Int =>x   #obj首先赋给x,然后判读x是否为Int类型
    case s:String =>s
    case s2:Double if(s2>0) =>s2  #if条件对double类型匹配进行限制
    case _:BigInt =>Int.MaxValue
    case _ =>0
    }
    ```
3. 数组匹配
    ```scala
    val sign = obj match{
    case Array(0) => "0"  #匹配包含0的数组
    case Array(x,y) => (x,y) #匹配任意两个元素的数组
    case Array(0, _*) =>'0...' #配置以0开始的数组
    case _ => 'else'
    }
    ```
4. 匹配列表
    ```scala
    case 0::Nil #包含0的List
    case x::y::Nil  #两个元素的列表
    case 0::tail  #以0开始的列表
    ```
5. 匹配元祖
    ```scala
    case (0，_) #以0开始的元祖
    case (x,0)  #以0结尾的两个元素元祖
    ```

### 1.7 case 样例类
- case class, 类似python的dataclass /命名元祖
- 可new 可不new
- 参数默认有getter、setter方法，参数类型默认为val,类外可访问
- 重写了equals,toString,hashCode等方法 
- 自动创建一个object伴生对象，该伴生对象包含apply,unapply方法
- 用于模式匹配和值存储(可以new，可以不new)
    ```scala
    abstract class Amount
    case class Dollar(value:Double) extends Amout
    case class Currency(value:Double) extends Amout
    def fun(amout:Amout)=amount match{
    case Dollar(v)=>'$‘+v
    case Currentcy(v)=>'RMB'+v
    }
    
    case class Dollar(value:Double)
    object Dollar{   # 自动创建的伴生对象
        def apply(value:Double)= new Dollar(value)
    }
    val dollar = Dollar(12.0) //不new，实际调用的伴生对象
    val dollar2 = new Dollar(12.0) //new, 调用的class
    ```

### 1.8 sealed和Option
- sealed class 密闭类，子类的种类有限（超类申明为sealed,子类和超类定义在一个文件)

    ```scala
    #可作为Enumrator的替代
    sealed trait Color // 声明color为密闭类 所有子类必须在同一个源文件,本身成员变量和方法
    case object Red extends Color  //case 对象
    case object Green extends Color
    case object Blue extends Color
    
    object Colors {
        val values = Vector(Red,Green,Blue)
    }
    
    def display(c:Color)={
        c match {
            case Red => s"it is $c"
            case Blue => s"it is $c"
            case Green=> s"it is $c"
        }
    }
    Colors.values.map(display)
    ```

    

- Option 类：表示可能存在也可能不存在的值
    ```scala
    val value:Option[String] = map.get(key) #key存在时value有值否者value为None
    def show(v:Option[String]) = v match{
    case Some(v) =>v  #用Some包裹存在的值
    case None =>'?'   #用None表示不存在值
    }
    
    //向现有类中添加新参数
    case class A(a:String, b:String, c:Option[String]=None)
    val a = class A("a","b")
    val b = class A("a","b",Option("c"))
    ```

### 1.9 隐式类型转换
- 隐式值： implict 关键字修饰， implict val name:String = "hello"
- 隐式参数： 方法中参数有implict修饰，隐式和非隐式混合是需要柯里化方式定义
    - def sayName(implicit name:String, age:int):Unit={} 默认所有参数都是隐式参数
    - def sayName(age:Int)(implict name:String):Unit={...} 只有name是隐式参数
- 方法调用时，不必手动参入方法中的隐式参数，scala会自动在当前作用域寻找隐式值传入
- 同类型的隐式值只能在当前作用域内出现一次（自动寻找不矛盾）

- 隐式转换函数（实现decortor类似的功能)
    - implict修饰的方法
    - 类型转换函数
    - 作用域内不能有相同参数类似和返回类型的
    - A类型没有method()，B类型有method(), 调用A.method()时编译器会检查作用域内有没有输入A类型返回B类型的隐式类型转换函数，如果有则A.method()可以调用
    - implict def AtoB(a:A):B={}

- 隐式类（实现decortor类似的功能)
    - implict class 
    - 必须定义在class 和object内部
    - 隐式类必须且只能传入一个类型参数
    - A.method()没有定义，则会在作用域和包内寻找接受A类型作为参数的类，而该类中实现了method()方法,则A.method()会调用成功
    - implicit class[a:A]:Unit={}


### 1.10 Actor机制
- 类似java中的线程
- 消息队列实现Actor之间的通信，异步非阻塞，避免繁琐的锁机制
- AKKa库，编写Actor模型应用(spark1.6)
- Netty库(spark2.0)
    ```scala
    import scala.actors.Actor  
    class MyActor extends Actor{}
    val actor= new MyActor()   
    actor.start()  
    actor ! "hello word"  #发送消息
    ```



### 1.11  包

1. 包的导入

   - import packagename.ClassName
- import util.Random
   - import util.{Random, Property}  // type newName = Random 为Random取第二个名字
   - import util.{Random=>Bob, Property=>Alice}  为包取别名 
   - import util._    //_导入包下的所有内容 
   - import java.util.Date // 可以直接导入java包

2.  包的创建

   1. 第一步创建包文件

   ```scala
   // filenam.scala
   package packgename  # 第一行的非注释语句
   def fun:={}
   class A
   ```

   2. 编译包文件

   ```shell
   scalac filename.scala
   ```

   3.  CLASSPATH 中添加编译包的路径

### 1.12 异常

```scala
class Problem(val msg:String) extends Exception

def f(n)={
    if (n==0){
        throw new Problem('input is zero')
    }
    else{
        n/12
    }
}

def test(n:Int)={
    try{ 
        f(n)
    }
    catch{
        case e:Problem=>s" Failed $e.msg"
        case _=>"unkown error"
    }
}
```

