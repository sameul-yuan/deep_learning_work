#! -*- coding:utf-8 -*- 

class MinHeap(object):
    """min heap for element of type(key,val)"""
    def __init__(self,max_size=None,compare_key=lambda x: x):
        self.max_size = max_size
        self.array =[None]*max_size 
        self.fn = compare_key
        self._count = 0

    def show(self):
        if self._count<=0:
            print('null')
        print(self.array[:self._count],end=', ')
    
    def min(self):
        if self._count>0:
            return self.array[0]
        else:
            raise ValueError("empty heap!")
    
    def pop(self):
        if self._count<=0:
            raise Exception('pop from an empty heap')
        value = self.array[0]
        self._count -=1 
        self.array[0] = self.array[self._count]
        self._shift_down(0)

        return value 
    
    def add(self, val):
        delete =None 
        save = False 
        if self._count >= self.max_size:
            min_v = self.min()
            if self.fn(val) >self.fn(min_v):
                self.array[0] = val
                self._shift_down(0)
                delete=min_v
                save=True
        else:
            self.array[self._count] = val 
            self._shift_up(self._count)
            self._count += 1
            save = True 
        return delete, save 

    def _shift_up(self, index):
        #比较节点与父节点的大小， 较小的为根节点
        if index > 0: 
            parent = (index -1)//2
            if self.fn(self.array[parent]) > self.fn(self.array[index]):
                self.array[parent], self.array[index] = self.array[index],self.array[parent]
                self._shift_up(parent)
    
    def _shift_down(self,index):
        #index 是父节点
        if index < self._count:
            left = 2*index+1
            right = 2*index+2
            #判断左右节点是否越界，是否小于根节点，是则交换
            if left<self._count and right < self._count:
                pivot = left if self.fn(self.array[left]) < self.fn(self.array[right]) else right 
            elif left <self._count:
                pivot = left 
            else:
                return 
            
            if self.fn(self.array[index]) > self.fn(self.array[pivot]):
                self.array[index],self.array[pivot] = self.array[pivot],self.array[index]
                self._shift_down(pivot)
            

if __name__ == '__main__':
    import  numpy as np 
    heap = MinHeap(max_size=5)
    a = np.random.randint(0,10,10)
    b =['a','b','c','d','e','f','g','h']
    a  = list(zip(a,b))
    print(a)
    for i in a:
        heap.add(i)
        heap.show()
