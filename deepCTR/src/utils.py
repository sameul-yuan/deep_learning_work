from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ConfigParser():
    pass

def parser_yaml(yaml_config):
    config = ConfigParser()
    for k, v in yaml_config.items():
        if isinstance(v, dict):
            v = parser_yaml(v)
        setattr(config,k,v)
    return config


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def calc_threshold_vs_depth(y_true, y_prob, stats_file=None):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    # print(y_prob[:100])
    ns = len(y_true)
    index = np.argsort(y_prob)
    index = index[::-1]
    y_prob = y_prob[index]
    # print(y_prob[:100])
    ratios = [0.001,0.002,0.003,0.004,0.005, 0.01,0.05, 0.1,0.15, 0.2, 0.25,0.3,
               0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1]
    
    pos_num = sum(y_true)
    pv = pos_num/len(y_true)
    depths =[]
    rates =[]
    samples=[]
    covers =[]
    lifts=[]
    p_thresholds=[]
    for ratio in ratios:
        top_k = max(1,int(ns*ratio))
        index1 = index[:top_k]
        top_true =  y_true[index1] 
        hit_rate = sum(top_true)/top_k  
        cover = sum(top_true)/pos_num
        p_threshold = y_prob[top_k-1]
        lift = hit_rate/pv
        
        depths.append(ratio)
        rates.append(hit_rate)
        samples.append(top_k)
        covers.append(cover)
        lifts.append(lift)
        p_thresholds.append(p_threshold)
        
    df = pd.DataFrame({'深度':depths,'命中率': rates, '覆盖率':covers, '样本数':samples,
                  '提升度':lifts, '概率门限':p_thresholds})
    if stats_file is not None:
        df.to_csv(stats_file, encoding='utf-8')   
    return df

if __name__== "__main__":
    print('task test')
