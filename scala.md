# scala基本知识
## 1. scala的语法:
### 1.1 六大特性
1. 和java无缝集成，运行在JVM上，编译成.class字节码
2. 自动类型推断，val，var就可以
3. 支持分布式和并发Actor（java中的threading）
4. 特质： java中的抽象类和接口功能的集合
5. 模式匹配：switch语句中不仅可以匹配基础对象，还可以直接匹配类型
6. 函数式语言，函数能当值使用；
### 1.2 类和对象
- 当new类时，类中除了方法不执行其他都执行
- object相当于java中的单例，定义的都的静态的，没有括号传参,(可以实现apply方法实现调用时传参)
- class 可以传参（scala特有），传参就有默认的构造
- 重新构造时，第一行先调用默认的构造
- for 
    - 1 to 10
    - 1.to(10)
    - 1 until 10, 不包括10 
    - for(i<- 1 to 10)
- while..., do ... while..., i+=1,
    - if ....else...
### 1.3 方法和函数
1. 方法的定义
    - def func
    - 方法中传参需要指定类型
    - 方法可以自动推断返回类型，模型将最后一行计算的结果当做返回值
    - 如果要写return返回值，需显示的声明返回类型
    - 如果方法体可以一行搞定，方法体的{}可省略
    - 如果方法名和方法体直接没有= 则为过程，返回Unit 
2. 递归方法
    - 显示声明返回类型
3. 参数可以有默认值
    - `def fun(x:Int=100):Unit={}`
4. 可变长参数
    - `def fun(x:String*):Unit={}`
5. 匿名函数
    - `()=>{}`
    - 可以赋值给一个变量，供调用
6. 偏应用函数
7. 嵌套方法
8. 高阶函数
    - 参数是函数
    - 返回类型是函数 
    - 参数和返回都是函数 
9. 柯里化函数
    - 高阶函数的简化（返回值类型是函数)

### 1.4 String 
- 就是Java中的String 
- println(s"this is $name") ；python中f-string功能类似
### 1.5 集合
1. Array
2. ArrayBuffer 
3. List 
    - `val list = List[String](...)`
    - map
    - flatMap
4. ListBuffer 
5. Set 
    - mutable.Set ,imutableSet
6. Map
    - map中的元素就是一个个的(k,v) 元祖
    - mutable.Map, imutable.Map 
    - HashMap 
7. Tuple
    - 最多22个元素
    - val iter = tuple.productIterator
    - 二元祖中有swap翻转函数
### 1.5 Trait 
- 相当于Java中接口和抽象类结合
- 可以在Trait中定义方法实现或不实现，var/val都可以
- 不可以传参
- 类继承Trait 第一个关键字是extends，之后用with

### 1.6 模式匹配
- match ... case ...
- 既可以匹配值，也可以匹配类型
- 从上往下匹配，匹配上自动break
- 匹配过程中会有值的转换(如1.0 转成int 1)
- _ 最后的默认匹配，default 
- 模型匹配语句相当于一大行

### 1.7 样例类
- case class
- 可new 可不new
- 参数默认有getter、setter方法，对外可见
- 重写了equals,toString等方法 
- 自动生成一个伴生对象，该伴生对象自动生成apply,unapply模板代码

### 1.8 偏函数
- def fun:PartionlFunction[String,int]={case ...}
- PartionlFunction[匹配类型，返回类型]
- java中的switch case

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
    >import scala.actors.Actor  
    > class MyActor extends Actor{}
    > val actor= new MyActor()   
    > actor.start()  
    > actor ! "hello word"  #发送消息
     

一、变量申明
val a =1 值类型，定义后无法改变，推荐使用便于垃圾回收（不可变类型，如果后面没用会回收内存）
var a=1  可变变量，
val a:Int = 1 变量类型为Int，变量和函数的返回类型写在后面 val name:type
常用的数据类型 Byte,Char,Short, Int, Long,Float, Double, Boolean, BigInt, BigDecimal
+-*/% to/until等操作符实际上是方法; val a = 1 to 10 , val a = 1.to(10)
val result = for( i<- 1 to 1000 if(i<500) if (i%2==0)) yield i 
不带参数的Scala方法通常不使用圆括号，一般没有参数且不改变当前对象的方法不带圆括号


object：java中的单例，object中定义的全是静态的，Array，List等都是object
object Lesson{   #object不可以传参,除非定义了apply方法（传参时执行apply方法）； 如果是object Person,则class Person类是该object Person对象的伴生类，该Person对象的Person类是伴生对象，可以互相访问私有变量；
  def main(args:Array[String]):Unit={
      val p = New Person('zhangshan',20,'M') #scala中new class时，类中除了方法不执行(构造方法执行)，其他都执行
      Lession(10) #调用apply
  }
  
  def apply(x:Int):Unit={
    println("apply called")
  }
}

class：普通类,类中可以传参（java不行），传参时需要制定类型（便于类中自动类型推断），有了参数就有默认的构造，类中属性默认有getter和setter方法
class Person( xname:String,xage:Int){ # xname,xage只有Person可见，如果是Person(val/var xname:String ,xage:Int) 则xname在类外可访问
    val name = xname   #默认public访问级别
    private val age = xage  #设为private访问级别
    var gender = 'F'
    def  sayName():Unit={  #Unit表示为返回（void）
      println("hello world")
      }
    def this(xname:String, xage:Int, xgender:Char){
       this(xname, xage)    # 重写构造时，第一行必须先调用默认构造函数
       this.gender = gender 
     }
}

trait： java中抽象类和接口的功能综合，trait不可以传参数
trait 中可以实现方法体也可以不实现， 继承类中需要实现没实现的方法


数据类型
Int，short，Long，String，Null(具有唯一实例null)，None（Option的子类，另一个是Some)，Any(Scala中基础基类），AnyRef，Unit(其他语言的void),
Nothing(推断不出的类型),Nil(0个元素的List)
二、控制结构与函数
每行语句的后面会自动补全分号；
if/else 结构的值是之后表达式的值： var r = if(x>0) 0 else 1
{}块表达式最后一个表达式的值就是块返回值

while(n>1); 
for(i<- 1 to n)  包含n
for(i <- 1 until n) 不包含n
for(i<- 1 to n; j<- 1 to n) 用;分隔多个生成器
for(i<1 1 to n if i%2==0; j<- 1 to n) 每个生成器可以带一个或多个if的守卫

函数
如果方法体使用return返回值，必须制定函数返回类型
如果方法体没有return，则默认使用方法体最后执行的一行作返回，返回类型可省略进行自动推导
方法体传递参数也必须制定参数类型
def abs(x:Double=10):Int = if (x>=0) x else -x  #如果方法体可以一行搞定，{}可省略
递归方法必须显示指定返回类型
def sum(args: Int*)={ for (arg<-args) result+=arg} 可变长参数
def sum(args: Int*)={args.foreach(ele=>{println(ele)}} 可变长参数,匿名函数
sum(1 to 5:_*) 使用_*将区间转成参数序列
def sum1() {println('this')} 没有等号的为过程，无论最后一行计算结果是什么，返回值都是为Unit；
lazy val words=scala.io.Source.fromFile(path).mkString 申明lazy的val初始化被推迟

匿名函数  ()=>{}, var fun：String=>Unit = (a:String)=>{println(a)}
                 def fun: (int, int)=>int=(a:Int,b:Int)=>{a+b}
                 
偏应用函数(参数绑定)  def fun: String=>Unit=showLog(2020,_:String);  fun('a') = showLog(2020,'a')

高阶函数：
1、方法的参数是函数 def fun(f:(int,int)=>Int, s:String):Unit={body}
2、方法的返回是函数 def fun(s1:String,s2:String):(string,string)=>String={body},显示的指示函数的返回类似是函数(在返回的函数后面加 _ 可以不用指明）
3、参数和返回都是函数

柯里化函数(高阶函数简化)
def fun(a:Int,b:Int)(c:Int,d:Int):Int={
    a+b+c+d
 }

三、常见数据操作
1： String Array List,只有Array可以new，都是不可变
String就是java中的String

Array固定长度数据，ArrayBuffer长度可变数组(import scala.collection.mutalbe.ArrayBuffer)
提供初始值不用new
val nums = new Array[Int](10) 10个初始值为0的整形数组
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

val array = new Array[Array[int]](3)
val b =ArrayBuffer[Int]() 初始化一个空的可变长数组
b += 1
b += (1,2,3) 最加元素
b ++= Array(1,2,3) 追加集合
b.trimEnd(2) 移除最后2个元素
b.insert(pos, num1,num2)
b.toArray  转成定长
b.boBuffer 转成可变长度

val list=List[Int](1,2,3)
val list.count(ele=>{})
val list.filter(ele=>{})
val result = list.map(s=>{s+1})
val result2 = list.flatMap( )   
val list=ListBuffer[Int](1,2,3)
用()来访问元素

for(elem <- arr)来遍历元素
for(elem <- arr if ..) yield ..将原数组转变为新数组；

val set = Set[Int](1,2,3,1) #没有重复元素
val set2 = Set[Int](2,3,4)
val result = set&set2 ;(&交集，&~差集)

import scala.collections.mutable.Set #自动找最近定义的
val set = Set[Int](1,2,3)  # val set = mutable.Set[Int](1,2,3)
set.+=(100)


四、常见的映射和元祖操作
映射->键值对集合

val score= Map[String,Int]('al'->90,'bl'->90,('c',90)) 默认是不可变的
val score = scala.collection.mutable.Map('al'->90)
score.put('a3',100)
val score = new scala.collection.mutable.HashMap[String,Int] 构造空映射，需要选定映射实现
score(key)  取值
score.get(key).get  #如果key存在通过get获取返回的Some内容
score.get(key).getOrElse(100)
score.getOrElse('al',defualt) 
score.contains(key)
score +=('sl'->90,'tl'->90) 添加映射
score -='sl'  移除键值对
for((k,v)<- score) 遍历
map1.++:(map2) # map1更新map2
map1.++(map2) # map2更新map1
map遍历时(k,v)组成元祖 tp._1,tp_2

元祖->对象(不一定相同类型)的集合，最多支持22个元素，可new可不new，可直接写元素
val (a,b,c)=(1,1,'a') 模型匹配获取元祖的值,
val res =(1,2,3)
res._1 ,res._2对应元祖中第一个和第二个元素

val tp1 = new Tuple1("hello")
val tp2 = new Tuple2("hello","that")

val iter:Iterator[Any] = tuple2.productIterator
iter.foreach(println)

tp1.toString

五、类定义
每个类都有一个主构造器，这个构造器和类的定义“交织”在一起，它的参数直接成为类的字段，主构造器执行类体中所有的语句
类中的字段自动带getter和setter方法,对于属性age, getter和setter方法分别为 def age(){} def age_={}
用@BeanProperty注解生成JavaBean的getXxx/setXxx方法
辅助构造器是可选的，它们都叫做this
class Counter(val arg :Int) {}

六、单例及伴生对象
可以用object语法定义结构，object定义了类的单个实例。
object不能定义构造器参数
高效地共享单个不可变实例
Scala没有静态方法或字段，可通过类和类同名的“伴生”对象来达到静态方法的目的
Scala程序必须从一个对象的main方法开始
object Main {
   def main(args:Array[String]):Unit={}
}
或者
object Main extends app{
    println()
}

七、继承
class A extends B{}
override def toString = super.toString  用override重写超类的非抽象方法,super调用超类方法
abstract class A定义抽象类，不能实例化
def sum:Int 抽象方式省去了方法体，子类中重写抽象字段、方法时不需要override
val att:Int 抽象字段没有初始值的字段

八、特质(trait)
scala用trait解决多重继承，类似java中的接口
trait中未实现的方法就想抽象的
class A extends trait1 with traint2{} with实现多个特质，后面的特质先执行

九、高阶函数

def fun(msg:String) {pirntln(msg)} 没有等号默认为过程
val f = fun _ 将函数赋给变量，_表示方法代表的这个函数

(x:Double)=> 3*x 匿名函数
val f = (x:Double)=>3*x 匿名函数赋给变量 def f = {(x:Double)=>3*x}
Array(1,2,3).map((x:Double)=>3*x)
Array(1,2,3).map((x)=>3*x) 由于map知道传入的类型，Double可去掉
Array(1,2,3).map(x=>3*x) 只有一个参数x，可省略()
Array(1,2,3).map(3*_) 参数在=>右侧只出现一次可用_代替

常见的高阶函数
map 
filter

柯里化：通过闭包的方法将接受两个参数的函数转换为接受一个参数的函数

十、模式匹配与样例类

match case 模式匹配 
匹配过程中会有数值转换
从上往下匹配，匹配上即终止


1）普通值匹配
val ch ='+'
val sign = ch match{
 case ‘+’ =>1
 case '-' => -1
 case parm if Character.isDigit(parm) => Character.digit(ch,10)  case后接变量则ch会赋给变量，用if给模式添加守卫
 case _ =>0      _代表default
}

2）类型匹配
val sign = obj match{
 case x:Int =>x   obj首先赋给x,然后判读x是否为Int类型
 case s:String =>s
 case _:BigInt =>Int.MaxValue
 case _ =>0
}

3）数组匹配
val sign = obj match{
  case Array(0) => "0"  匹配包含0的数组
  case Array(x,y) => (x,y) 匹配任意两个元素的数组
  case Array(0, _*) =>'0...' 配置以0开始的数组
  case _ => 'else'
}
4)匹配列表
case 0::Nil 包含0的List
case x::y::Nil  两个元素的列表
case 0::tail  以0开始的列表
5）匹配元祖
case（0，_） 以0开始的元祖
case（x,0) 以0结尾的两个元素元祖

6）偏函数：不能带括号传参，只能匹配一个值返回一个值，如下匹配string返回int
def MyTest:PartionlFunction[String,Int]={
 case "abc"=>2
 case "a"=>1
 case _=>100
}

case class 样例类：用于模式匹配和值存储(可以new，可以不new)，重写了equals,toString,hashCode等方法
abstract class Amount
case class Dollar(value:Double) extends Amout
case class Currency(value:Double) extends Amout
def fun(amout:Amout)=amount match{
  case Dollar(v)=>'$‘+v
  case Currentcy(v)=>'RMB'+v
}

sealed class 密闭类，子类的种类有限（超类申明为sealed,子类和超类定义在一个文件)
Option 类：表示可能存在也可能不存在的值
val value:Option[String] = map.get(key) key存在时value有值否者value为None
def show(v:Option[String]) = v match{
   case Some(v) =>v  用Some包裹存在的值
   case None =>'?'   用None表示不存在值
}

十一、常见的集合操作
所有的集合都是iterable的
集合包含seq，set、map三大类
1）序列：如数组、列表可通过下标索引
2）set：如Set无须的， LinkedHashSet 保留插入顺序，SortedSet排序的
3）map: 如Map， SortedMap按键排序
+将元素添加到无须的Set
+：和:+分别将元素前向和后向添加到序列
++将两个机会拼接
-和--删除元素

列表：要么是Nil(空列表),要么是一个head元素+tail列表
      ::从给定的头尾创建列表， ::是右结合类别将从末端开始构建

	  
集合的常用操作方法：
1）映射
s.map(fun) 
s.flatMap(fun) 展平映射  (spark中rdd调用的map，flatMap就是scala中的实现)

2）化简
s.reduceLeft/Right(op)
s.foldLeft(init)(op) 提供了额外初始值的reduce
s.scanLeft/Right(op) 产生中间结果和最后的结果

print(List(1,2,4).reduceLeft(_ + _)

3)zip 和Python类型
a zip b
a.zipAll(b,default_a, defualt_b)



