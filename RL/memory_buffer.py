import random 
import numpy as np 
from collections import deque 

class SumTree(object):
    write = 0 
    def __init_(self,capacity):
        self.capacity = capacity 
        self.tree = np.zeros(2*capacity -1 )
        self.data = np.zeros(capacity, dtype=object)

    def _propatate(self,idx, change):
        parent = (idx-1)//2
        self.tree[parent] += change 
        if parent != 0 :
            self._propatate(parent,change)
    def _retrive(self,idx,s):
        left = 2*idx +1 
        right = left + 1
        if left>=len(self.tree):
            return idx 
        if s<=self.tree[left]:
            return self._retrive(left,s)
        else:
            return self._retrive(right, s-self.tree[left])
        
    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity -1 
        self.data[self.write] = data 
        self.update(idx, p)

        self.write += 1
        if self.write >=self.capacity:
            self.write = 0
        
    def update(self,idx,p):
        change = p - self.tree[idx]
        self.tree[idx] = p 
        self._propatate(idx, change)
    
    def get(self, s):
        idx = self._retrive(0, s)
        dataidx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataidx])

class MemoryBuffer(object):
    """experience reply buffer using a double-end deque or a sum-tree"""
    def __init__(self,buffer_size, with_per = False):
        if with_per:
            "Prioritized Experience Replay"
            self.alpha=0.5 
            self.epsilon = 0.01 
            self.bufer =SumTree(buffer_size)
        else:
            self.buffer = deque()
        self.count =0 
        self.with_per = with_per 
        self.buffer_size = buffer_size 
    
    def memorize(self, state,action, reward, done,new_state,error=None):
        """ save an experience to memory, optionally with its td-error""" 
        experience =(state, action, reward, done,new_state)
        if self.with_per:
            priority = self.priority(error[0])
            self.buffer.add(priority, experience)
            self.count+=1
        else:
            if self.count <self.buffer_size:
                self.buffer.append(experience)
                self.count+=1 
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def priority(self,error):
        return (error + self.epsilon)**self.alpha
    
    def size(self):
        return self.count
    
    def sample_batch(self, batch_size):
        batch =[]

        if self.with_per:
            T = self.buffer.total()//batch_size 
            for i in range(batch_size):
                a,b = T*i, T*(i+1)
                s = random.uniform(a,b)
                idx, error, data = self.buffer.get(s)
                batch.append((*data,idx))
            idx = np.array([i[5] for i in batch])
        else:
            idx = None 
            batch = random.sample(self.buffer, min(batch_size,self.count))
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        ns_batch = np.array([i[4] for i in batch])

        return s_batch, a_batch, r_batch, d_batch, ns_batch

    def update(self,idx, new_error):
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        if self.with_per:
            self.buffer = SumTree(self.buffer_size)    
        else:
            self.buffer = deque()
        self.count = 0 
        