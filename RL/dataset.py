#! encoding: utf-8
import os 
import numpy as np 
import pandas as pd 
from itertools import cycle 

class CsvBuffer(object):
    def __init__(self,file_dir=None, reg_pattern=r'.csv',chunksize=32,**kwargs):
        self.files = None 
        self.iterator =None 
        self.is_buffer_available = False 
        self._chunksize = chunksize 
        self._csv_kwargs = kwargs 
        self._file_producer = self._file_pipe_producer(file_dir, reg_pattern)
        self._buffer_warmup()

    @staticmethod
    def _fetch_files(dir):
        if not os.path.exists(dir):
            raise ValueError("dir not exist")
        file_or_dirs = [os.path.join(dir,base) for base in os.listdir(dir)]
        files =[]
        for file in file_or_dirs:
            if os.path.isdir(file):
                _files = __class__._fetch_files(file)
                files.extend(_files)
            else:
                files.append(file)
        return files 
    
    def _file_pipe_producer(self,file_dir, pattern=None):
        import re 
        file_dir = os.path.abspath(file_dir)
        files = __class__._fetch_files(file_dir)
        if pattern:
            pattern = re.compile(pattern)
            files =[f for f in files if re.search(pattern,os.path.basename(f)) is not None]
        self.files = files 
        return cycle(self.files)

    def _buffer_warmup(self):
        file =next(self._file_producer)
        print('load data from {}'.format(file).center(50,'-'))
        self.iterator = pd.read_csv(file, chunksize=self._chunksize, **self._csv_kwargs)
        self.is_buffer_available = True 
    def __iter__(self):
        return self 

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self._buffer_warmup()
            data = next(self.iterator)
        return data.values

class Dataset(object):
    def __init__(self, x, batch_size=32, shuffle=True):
        self.X = np.asarray(x)
        self.batch_size = batch_size 
        self.size, self.dim = self.X.shape[0], self.X.shape[1]
        self.begin = 0
        self.shuffle = shuffle 
        self.index = np.arange(self.size)

    def __iter__(self):
        return self 
    
    def __next__(self):
        if (self.begin + 0.2*self.batch_size)>=self.size :
            self.begin = 0
            if self.shuffle: 
                print("shuffle dataset".center(50,'-')+'\n')
                self.index = np.random.permutation(self.size)
        self.end = min(self.begin + self.batch_size, self.size)
        index = self.index[self.begin:self.end]
        batch_data = self.X[index]
        self.begin += self.batch_size
        return batch_data
         




