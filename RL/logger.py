import os 
import time
import json 
import atexit
import numpy as np 

color2num=dict(gray=30, red=31,green=32,yellow=33,blue=34,magenta=35,cyan=36,white=37,crimson=38)
def colorize(string, color, bold=False, highlight=False):
    attr=[]
    num = color2num[color]
    if highlight:
        num +=10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' %(';'.join(attr),string)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True 
    except:
        return False 

def convert_json(obj):
    if is_json_serializable(obj):
        return obj 
    else:
        if isinstance(obj, dict):
            return {convert_json(k):convert_json(v) for k,v in obj.items()}
        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)
        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]
        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)
        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k):convert_json(v) for k,v in obj.__dict__.items()}
            return {str(obj):obj_dict}
        return str(obj)

class Logger(object):
    def __init__(self, output_dir=None, output_fname='progress.txt',exp_name=None):
        self.output_dir = output_dir or './log/experiments%d'%int(time.time())
        if os.path.exists(self.output_dir):
            print('warning: Log dir {} already exists! storing there anyway'.format(self.output_dir))
        else:
            os.makedirs(self.output_dir)
        
        self.output_file = open(os.path.join(self.output_dir, output_fname),'w+')
        atexit.register(self.output_file.close)

        print('loading data to {}'.format(self.output_file.name))

        self.first_row =True
        self.log_headers =[]
        self.log_current_row ={}
        self.exp_name = exp_name

    def log(self,msg,color='green'):
        print(colorize(msg,color,bold=True))
    
    def log_tabular(self,key,val):
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers,"trying to introduce a new key {} you didn't include in the first iteration".format(key)
        assert key not in self.log_current_row,"already set {} for this iteration. mayby you forgot to call dump_tabular()".format(key)
        self.log_current_row[key] = val
    
    def save_config(self,config):
        config_json =convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name']=self.exp_name
        output = json.dumps(config_json, separators=(',',':\t'),indent=4, sort_keys=True)
        print(colorize('saving config:\n',color='cyan',bold=True))
        with open(os.path.join(self.output_dir,"config.json"),'w') as out:
            out.write(output)

    def dump_tabular(self, stdout=False):
        vals =[]
        if stdout:
            max_key_len = max(map(len,self.log_headers))
            max_key_len = max(15, max_key_len)
            keystr= "%" + '%d'%max_key_len
            fmt = "| " +keystr + "s | %15s"
            n_slashes = 22 + max_key_len
            print("-"*n_slashes)
        
        for key in self.log_headers:
            val = self.log_current_row.get(key,np.nan)
            valstr = "%.4f"%val if hasattr(val, "__float__") else val 
            vals.append(valstr)
        
        if stdout:
            for key,valstr in zip(self.log_headers, vals):
                print(fmt %(key,valstr))
            print("-"*n_slashes)
        
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+'\n')
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False 

if __name__=='__main__':
    logger = Logger(output_dir='../assets',output_fname='log.txt')
    logger.log('test',color='green')
    logger.save_config(locals())
    logger.log_tabular('a',1)
    logger.log_tabular('b',2)
    print('----')
    logger.dump_tabular()

        

         
        