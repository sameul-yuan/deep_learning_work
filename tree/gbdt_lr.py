from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import pandas as pd
import os
import numpy as np

X,y = make_classification(n_classes=2, n_samples=2000)

train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.1, random_state=42, stratify=y)


dtrain = lgb.Dataset(train_x, label=train_y,categorical_feature=None)
dval = lgb.Dataset(test_x, label=test_y, reference=dtrain,categorical_feature=None)

lgb_config = {
            "learning_rate": 0.021501026995007368,
            "max_depth": 6,
            "num_leaves": 28,
            "min_data_in_leaf": 30,
            "bagging_fraction": 0.7256064804736237,
            "bagging_freq": 1,
            "feature_fraction": 0.5542570143757172,
            "min_gain_to_split": 3.6066347929789426,
            "min_data_in_bin": 10,
            "lambda_l2": 0.5669433809588953,
            "lambda_l1": 0.457594518638886,
            "seed": 42,
            "num_threads": -1,
            "min_sum_hessian_in_leaf": 0.001,
            "max_cat_threshold": 32,
            'fobj': None,
            "feval": None,
            # "learning_rates": lambda x: 0.002 if x<5 else 0.03*exp(-floor(x/200)),#[0,n_iters)
            "learning_rates": None,
            "objective": "binary",
            "is_unbalance": False,
            'zero_as_missing': False,
            "metric": ["binary_logloss", "auc"],
            "metric_freq": 5,
            "boosting": "gbdt",
            "verbose": 0,
            'boost_from_average': True}  # default True

num_boost_round = 10
evalists =[dtrain, dval]
evalnames=['train','val']
early_stopping_rounds =40
eval_result={}
model = lgb.train(lgb_config, dtrain, num_boost_round, evalists, evalnames,
                  early_stopping_rounds=early_stopping_rounds,
                  verbose_eval=20, evals_result=eval_result,
                  feval=None, fobj=None,
                  learning_rates=None)

train_pred=model.predict(train_x,pred_leaf=True)
test_pred= model.predict(test_x,pred_leaf=True)

# df = pd.DataFrame(train_pred).astype(int)
# uniques = df.apply(lambda x: pd.unique(x).tolist())
# print(uniques)

test_pred_prob = model.predict(test_x, predict_prob=True)
score = roc_auc_score(test_y, test_pred_prob)

print('score', score)
print(train_pred)
print(train_pred.shape)
tree_leafs = train_pred.max(axis=0)+1
print(tree_leafs)
cumsum_leafs = tree_leafs.cumsum()

lr_train_input = np.zeros((train_pred.shape[0], tree_leafs.sum(axis=0)))
lr_test_input = np.zeros((test_x.shape[0], tree_leafs.sum(axis=0)))
for i, el in enumerate(train_pred):
    set_index =[leaf_index if i==0 else leaf_index +cumsum_leafs[i-1] for i, leaf_index in enumerate(el)]
    lr_train_input[i, set_index] = 1

for i, el in enumerate(test_pred):
    set_index = [leaf_index if i==0 else leaf_index+cumsum_leafs[i-1] for i, leaf_index in enumerate(el)]
    lr_test_input[i, set_index] =1

print(lr_train_input)
print(lr_train_input.shape)

lr_model = LogisticRegression()
lr_model.fit(lr_train_input, train_y)
lr_test_pred = lr_model.predict_proba(lr_test_input)[:,1]
score = roc_auc_score(test_y, lr_test_pred)
print('score',score)







