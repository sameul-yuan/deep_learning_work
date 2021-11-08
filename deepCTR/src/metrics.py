import os
import pickle
from functools import reduce

from tqdm import tqdm

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score, plot_precision_recall_curve

import matplotlib.pyplot as plt

def optimal_binning_boundary(x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
    '''
        利用决策树获得最优分箱的边界值列表
    '''
    boundary = []  # 待return的分箱边界值列表
    
    x = x.fillna(nan).values  # 填充缺失值
    y = y.values
    clf = DecisionTreeClassifier(criterion='entropy',    #“信息熵”最小化准则划分
                                 max_leaf_nodes=6,       # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比
    clf.fit(x.reshape(-1, 1), y)  # 训练决策树
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])
    boundary.sort()
    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]
    return boundary


def ks_table(y_predict, y_true, ascending=False, is_cate=False, sep=None, nbins=10):
    """
    计算KS分层表
    :param y_predict: list, array, pandas.Series, 预测的概率或打分
    :param y_true: list, array, pandas.Series, 真实的标签, 1或0, 只支持二分类
    :param ascending: boolean, default False, 对y_predict排序的方式。
        为False则降序排序,此时y_predict越大表示y_true为1(坏客户)的概率越大,一般适用于预测概率;
        为True则升序排序,此时y_predict越小表示y_true为1(坏客户)的概率越大,一般适用于标准分;
    :param sep: list, default None, 预设的分割点
    :return: Pandas.DataFrame, 结果KS分层表
    """
    if len(y_predict) < 10:
        return None
    if not isinstance(y_predict, pd.Series):
        y_predict = pd.Series(y_predict)
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    y_predict = y_predict.reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)
    data = pd.concat([y_predict, y_true], axis=1).reset_index(drop=True)
    data = data.dropna()
    data.columns = ['score', 'flag']
    data.sort_values(by='score', ascending=ascending, inplace=True)
    data = data.reset_index(drop=True)
    cnt_all = data.shape[0]

    if not is_cate:   
        if sep is not None:
            sep = sorted(sep)
            if ascending:
                data["group"] = pd.cut(data['score'], bins=sep, labels=False)
            else:
                data["group"] = pd.cut(data['score'], bins=sep, labels=False).map(lambda x: len(sep) - 1 - x)
            nbins = len(sep) - 1
        else:
            data["group"] = data.index.map(lambda x: min(nbins - 1, (x + 1) // (cnt_all / nbins)))
            sep = data.groupby("group").agg({"score": "max"})["score"].tolist()
            if ascending:
                sep[-1] = float("inf")
                sep = [float("-inf")] + sep
            else:
                sep[0] = float("inf")
                sep.append(float("-inf"))
    else:
        data['group'] = data['score']
        nbins = data['score'].nunique()

    table = pd.DataFrame(np.arange(1, nbins + 1), columns=['Rank'])
    table['Total'] = data.groupby('group').size().reindex(table.index).fillna(0)  # Total这一列存放每一组样本的总个数
    table['Population'] = table['Total'] / cnt_all
    table['MinScore'] = data[['score', 'group']].groupby(['group']).min()
    table['MeanScore'] = data[['score', 'group']].groupby(['group']).mean()
    table['MaxScore'] = data[['score', 'group']].groupby(['group']).max()
    table['BadCnt'] = data[['flag', 'group']].groupby(['group']).sum().reindex(table.index).fillna(0)
    table['bumps'] = 0
    table['GoodCnt'] = table['Total'] - table['BadCnt'].fillna(0)
    table['InnerBadRate'] = table['BadCnt'] / table['Total']
    table['bumps'] = 0
    for ind in range(1, table.shape[0]):
        if table.loc[table.index[ind], 'InnerBadRate'] > table.loc[table.index[ind - 1], 'InnerBadRate']:
            table.loc[table.index[ind], 'bumps'] = table.loc[table.index[ind - 1], 'bumps'] + 1
        else:
            table.loc[ind, 'bumps'] = table.loc[ind - 1, 'bumps']
    table['bumps'] = table['bumps'].astype('int64').fillna(0)
    table['BadRate'] = table['BadCnt'] / sum(table['BadCnt'])
    table['CumTotalBadRate'] = table['BadRate'].cumsum()
    table['GoodRate'] = table['GoodCnt'] / sum(table['GoodCnt'])
    table['CumTotalGoodRate'] = table['GoodRate'].cumsum()
    table['K-S'] = (table['CumTotalBadRate'] - table['CumTotalGoodRate'])
    table['Lift'] = table['InnerBadRate'] * (table['BadCnt'].sum() + table['GoodCnt'].sum()) / table['BadCnt'].sum()

    table['BadRate'] = table['BadRate'].apply(lambda x: format(x, '.2%'))
    table['CumTotalBadRate'] = table['CumTotalBadRate'].apply(lambda x: format(x, '.2%'))
    table['CumTotalGoodRate'] = table['CumTotalGoodRate'].apply(lambda x: format(x, '.2%'))
    total_information = {'Rank': 'Total', 'Population': 1.0, 'Total': data.shape[0], 'MinScore': min(y_predict), 'MeanScore': np.mean(y_predict),
                         'MaxScore': max(y_predict), 'BadCnt': sum(table['BadCnt']), 'GoodCnt': sum(table['GoodCnt']),
                         'InnerBadRate': sum(table['BadCnt']) / len(data), 'bumps': '.', 'BadRate': '.', 'CumTotalBadRate': '.', 'CumTotalGoodRate': '.',
                         'GoodRate': sum(table['GoodCnt']) / len(data), 'K-S': max(table['K-S']), 'Lift': '.'}
    table = table.append(total_information, ignore_index=True)
    selected_columns = ['Rank', 'Population', 'Total', 'bumps', 'MinScore', 'MeanScore', 'MaxScore', 'BadCnt', 'GoodCnt', 'InnerBadRate', 'BadRate',
                        'CumTotalBadRate', 'GoodRate', 'CumTotalGoodRate', 'K-S', 'Lift']
    table = table.loc[:, selected_columns]
    table['InnerBadRate'] = table['InnerBadRate'].apply(lambda x: format(x, '.2%'))
    table['GoodRate'] = table['GoodRate'].apply(lambda x: format(x, '.2%'))
    table['K-S'] = table['K-S'].apply(lambda x: format(x, '.2%'))
    table['Population'] = table['Population'].apply(lambda x: format(x, '.2%'))
    table['Lift'] = table['Lift'].map(lambda x: round(x, 2) if x != '.' else 1)
    table['MinScore'] = table['MinScore'].apply(lambda x: format(x, '.2f'))
    table['MaxScore'] = table['MaxScore'].apply(lambda x: format(x, '.2f'))
    table['MeanScore'] = table['MeanScore'].apply(lambda x: format(x, '.2f'))
    return table, sep


def iv_table(x: pd.Series, y: pd.Series, is_cate: bool = False, nan: float = -999.) -> pd.DataFrame:
    '''
        计算变量各个分箱的WOE、IV值，返回一个DataFrame
    '''
    #x = x.fillna(nan)
    
    df = pd.concat([x, y], axis=1)                        # 合并x、y为一个DataFrame，方便后续计算
    df.columns = ['x', 'y']                               # 特征变量、目标变量字段的重命名
    
    stat_num = df['y'].value_counts()
    good_num,bad_num = stat_num[0],stat_num[1]
    total_num = good_num+bad_num
    
    df = df.dropna()
    if not is_cate:
        boundary = optimal_binning_boundary(df.x, df.y, nan)        # 获得最优分箱边界值列表
        df['bins'] = pd.cut(x=x, bins=boundary, right=False)  # 获得每个x值所在的分箱区间   
    else:
        df['bins'] = df['x']

    grouped = df.groupby('bins')['y']                     # 统计各分箱区间的好、坏、总客户数量
    result_df = grouped.agg([('good',  lambda y: (y == 0).sum()), 
                             ('bad',   lambda y: (y == 1).sum()),
                             ('total', 'count')])
    result_df['good_pct'] = result_df['good'] / result_df["good"].sum()       # 好客户占比
    result_df['bad_pct'] = result_df['bad'] / result_df["bad"].sum()          # 坏客户占比
    result_df['goodRecall'] = result_df['good'] / good_num       # 好客户占比
    result_df['badRecall'] = result_df['bad'] / bad_num         # 坏客户占比
    result_df['groupRecall'] = result_df['total'] / total_num    # 总客户占比
    result_df['innerBadRate'] = result_df['bad'] / result_df['total']             # 坏比率
    result_df['woe'] = np.log(result_df['good_pct'] / result_df['bad_pct'])             # WOE
    result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']  # IV
    result_df['lift'] = (result_df['bad']/result_df['total'])/(bad_num/total_num)  # lift
    result_df = result_df.reset_index()
    result_df.sort_values("bins",ascending=False,inplace=True)
    result_df['bins'] = result_df['bins'].astype('str')
    total_information = {'bins': 'Total', 'good': result_df['good'].sum(), 'bad': result_df['bad'].sum(), 'total': result_df['total'].sum(), 'good_pct': result_df['good_pct'].sum(),
                         'bad_pct': result_df['bad_pct'].sum(), 'goodRecall': result_df['goodRecall'].sum(), 'badRecall': result_df['badRecall'].sum(),
                         'groupRecall': result_df['groupRecall'].sum(), 'innerBadRate': result_df["bad"].sum()/(result_df["good"].sum()+result_df["bad"].sum()), 
                         'woe': '.','iv': result_df['iv'].sum(), 'lift': (result_df["bad"].sum()/(result_df["good"].sum()+result_df["bad"].sum()))/(bad_num/total_num)}
    result_df = result_df.append(total_information, ignore_index=True)
    
    return result_df


def df_auc(x: pd.Series, y: pd.Series, ifnan:bool=False, nan: float = -999.) -> float:
    '''计算基于dataframe数据的auc'''
    df = pd.concat([x, y], axis=1)
    df.columns = ['x', 'y']
    if ifnan:
        df = df.fillna(nan)
    else:
        df = df.dropna()
    return roc_auc_score(df.y, df.x)


def score_summary(x: pd.Series, y: pd.Series, is_cate: bool = False, ifnan:bool=False, nan: float = -999.) -> dict:
    summary_dict = {'iv':None, 'auc':None, 'coverage_rate':None}
    
    df = pd.concat([x, y], axis=1)
    df.columns = ['x', 'y']
    
    # 1、覆盖率
    coverage_rate = round(1-df.x.isna().sum()/len(df), 4)
    summary_dict['coverage_rate'] = coverage_rate
    
    if ifnan:
        df = df.fillna(nan)
    else:
        df = df.dropna()
    
    # 2、ks相关    
    ks_t = ks_table(df.x, df.y, is_cate=is_cate, ascending=False, sep=None)
    summary_dict['ks'] = float(ks_t[0]['K-S'].iloc[-1].strip('%'))
    
    #auc相关
    auc = df_auc(df.x, df.y, ifnan=False)
    summary_dict['auc'] = round(auc,4)
    
    # 3、iv相关
    iv_t = iv_table(df.x, df.y, is_cate=is_cate)
    summary_dict['iv'] = iv_t['iv'].iloc[-1]
    
    return iv_t, ks_t[0], summary_dict


class Psi(object):
    def __init__(self, bins: int = 10, minimal: int = 1):
        self.bins = bins
        self.minimal = minimal
        self.psi_detail = dict()
        self.psi = pd.DataFrame()
        self.base = dict()

    def _distribution_continuous(self, series: pd.Series, bins: list = None):
        if bins:
            bins[0] = float("-inf")
            bins[-1] = float("inf")
            series_cut = pd.cut(series, bins=bins)
        else:
            try:
                _, bins = pd.qcut(series, q=self.bins, retbins=True, duplicates="drop")
                if len(bins) == 1:
                    bins = [float("-inf")] + bins.tolist() + [float("inf")]
            except IndexError:
                bins = [float("-inf"), float("inf")]
            bins[0] = float("-inf")
            bins[-1] = float("inf")
            series_cut = pd.cut(series, bins=bins)
        series_cut.cat.add_categories("_missing", inplace=True)
        series_cut = series_cut.fillna("_missing")
        return series_cut, list(bins)

    @staticmethod
    def _distribution_discrete(series: pd.Series, values: set = None):
        flag = 0
        if isinstance(series.dtypes, pd.core.dtypes.dtypes.CategoricalDtype):
            series.cat.add_categories("_missing", inplace=True)
            series = series.fillna("_missing")
            flag = 1
        if series.dtypes in ("int", "float"):
            values = ["_missing"] + [str(i) for i in sorted(series.dropna().unique(), key=lambda x: float(x))]
        if values:
            series_trans = series.map(lambda x: str(x) if str(x) in values else "_missing")
        else:
            series_trans = series.astype("str").fillna("_missing")
            if flag == 1:
                values = sorted(series_trans.unique(), key=lambda x: float(x.split(", ")[1][:-1] if x != "_missing" else float("-inf")))
            else:
                values = sorted(series_trans.unique())
            if "_missing" not in values:
                values = ["_missing"] + values
        return series_trans, values

    def _psi(self, all_series: list, names: list):
        has_nan = reduce(lambda a, b: a or b, [x.loc["_missing"] for x in all_series])
        if not has_nan:
            all_series = [series.drop(labels="_missing") for series in all_series]
        base = all_series[0]
        all_series = [base.replace(0, self.minimal)] + [series.reindex(base.index).replace(0, self.minimal).fillna(self.minimal) for series in
                                                        all_series[1:]]
        all_series = [series / series.sum() for series in all_series]
        res = pd.DataFrame({names[i]: all_series[i] for i in range(len(names))})
        psi = {names[0]: 0}
        psi.update(
            {names[i + 1]: ((all_series[0] - all_series[i + 1]) * np.log(all_series[0] / all_series[i + 1])).sum() for i in range(len(names) - 1)})
        res.index = res.index.astype("category").add_categories("psi")
        res.loc["psi"] = psi
        return res

    def fit(self, df: pd.DataFrame, part_column: str, continuous_columns: list = (), discrete_columns: list = (), part_values: list = None,
            priori: dict = None):
        continuous_columns = list(continuous_columns)
        discrete_columns = list(discrete_columns)
        df = df.reset_index(drop=True)
        part = df[part_column]
        all_parts = part_values or sorted(part.unique())
        indexes = [part[part == value].index for value in all_parts]
        max_length = max([len(i) for i in continuous_columns + discrete_columns]) + 35
        p_bar = tqdm(continuous_columns)
        for col in p_bar:
            p_bar.set_description(f"Processing continuous features {col}".ljust(max_length, " "))
            all_series = [df[col].loc[idx] for idx in indexes]
            if priori is None:
                tmp_base = all_series[0]
                tmp_base_trans, tmp_bins = self._distribution_continuous(tmp_base)
                all_series = [tmp_base_trans] + [self._distribution_continuous(series, tmp_bins)[0] for series in all_series[1:]]
                all_series = [series.value_counts(sort=False) for series in all_series]
                self.base[col] = [all_series[0], tmp_bins]
                tmp_psi = self._psi(all_series, all_parts)
            else:
                tmp_base_trans, tmp_bins = priori[col]
                all_series = [self._distribution_continuous(series, tmp_bins)[0] for series in all_series]
                all_series = [tmp_base_trans] + [series.value_counts(sort=False).sort_index() for series in all_series]
                tmp_psi = self._psi(all_series, ["base"] + all_parts)
            self.psi_detail[col] = tmp_psi
            tmp_psi_summary = {"var": col}
            tmp_psi_summary.update(tmp_psi.loc["psi"].to_dict())
            self.psi = self.psi.append(tmp_psi_summary, ignore_index=True)
        p_bar = tqdm(discrete_columns)
        for col in p_bar:
            p_bar.set_description(f"Processing discrete features {col}".ljust(max_length, " "))
            all_series = [df[col].loc[idx] for idx in indexes]
            if priori is None:
                tmp_base = all_series[0]
                tmp_base_trans, tmp_values = self._distribution_discrete(tmp_base)
                all_series = [tmp_base_trans] + [self._distribution_discrete(series, tmp_values)[0] for series in all_series[1:]]
                all_series = [series.value_counts(sort=False).reindex(tmp_values).fillna(0) for series in all_series]
                self.base[col] = [all_series[0], tmp_values]
                tmp_psi = self._psi(all_series, all_parts)
            else:
                tmp_base_trans, tmp_values = priori[col]
                all_series = [self._distribution_discrete(series, tmp_values)[0] for series in all_series]
                all_series = [tmp_base_trans] + [series.value_counts(sort=False).sort_index() for series in all_series]
                tmp_psi = self._psi(all_series, ["base"] + all_parts)
            self.psi_detail[col] = tmp_psi
            tmp_psi_summary = {"var": col}
            tmp_psi_summary.update(tmp_psi.loc["psi"].to_dict())
            self.psi = self.psi.append(tmp_psi_summary, ignore_index=True)
        self.psi = self.psi.set_index("var")
        if priori is not None:
            self.psi = self.psi[["base"] + all_parts]
        else:
            self.psi = self.psi[all_parts]
        self.psi.columns = ["psi_" + str(col) for col in self.psi.columns]

    def save_base(self, path):
        pickle.dump(self.base, open(path, "wb"))
