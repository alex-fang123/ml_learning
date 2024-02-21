import pandas as pd
import numpy as np


def get_information_entropy(data, kind_tag):
    """
    计算信息熵
    :param data: 原数据集
    :param kind_tag: 标记类别的列
    :return: 数据集的信息熵
    """
    temp = data.groupby(kind_tag)
    p = temp.size() / data.shape[0]
    return -np.sum(p * np.log2(p))


def get_information_gain_discrete(data, kind_tag, tar_feature):
    """
    计算信息增益
    :param data: 原数据集
    :param kind_tag: 标记类别的列
    :param tar_feature: 目标特征
    :return: 信息增益
    """
    Ent_total = get_information_entropy(data, kind_tag)
    split_data = data.groupby(tar_feature)
    for [i, item] in split_data:
        Ent_total -= get_information_entropy(item, kind_tag) * item.shape[0] / data.shape[0]
    return Ent_total


def get_information_gain_continuous(data, kind_tag, tar_feature):
    # 连续属性的信息增益计算
    feature_values = data[tar_feature].unique()
    feature_values.sort()
    split_point = (feature_values[:-1] + feature_values[1:]) / 2
    result_point = None
    max_gain = 0
    for point in split_point:
        Ent_total = get_information_entropy(data, kind_tag)
        left_data = data[data[tar_feature] <= point]
        right_data = data[data[tar_feature] > point]
        Ent_total -= get_information_entropy(left_data, kind_tag) * left_data.shape[0] / data.shape[0]
        Ent_total -= get_information_entropy(right_data, kind_tag) * right_data.shape[0] / data.shape[0]
        if Ent_total > max_gain:
            max_gain = Ent_total
            result_point = point
    return max_gain, result_point


def get_the_best_split_feature(data, continuous_feature, kind_tag):
    """
    获取最佳分裂属性
    :param data: 数据集
    :param continuous_feature: 连续属性集合
    :return: 最佳划分属性
    """
    feature_num = data.shape[1] - 2
    gain = pd.DataFrame(columns=data.columns.tolist()[1:-1])
    split_point = pd.DataFrame(columns=data.columns.tolist()[1:-1])
    gain.loc[0] = np.zeros(feature_num)
    split_point.loc[0] = np.zeros(feature_num)
    column_names = data.columns.tolist()
    for i in range(1, feature_num + 1):
        if column_names[i] not in continuous_feature:  # 离散属性值
            gain.iloc[0, i - 1] = get_information_gain_discrete(data, kind_tag, column_names[i])
            split_point.iloc[0, i - 1] = None
        else:  # 连续属性值
            [gain.iloc[0, i - 1], split_point.iloc[0, i - 1]] = get_information_gain_continuous(data, kind_tag,
                                                                                                column_names[i])
    split_feature = gain.idxmax("columns").tolist()[0]
    return split_feature, split_point


def GenerateTree(data, kind_tag, continuous_feature):
    """
    生成决策树
    :param data: 原数据集
    :return: 决策树，是一个嵌套的字典
    """
    if data[kind_tag].nunique() == 1:  # 如果集合中的样例全部属于同一类，如全部为正例或者全部为反例
        return data[kind_tag].unique()[0]
    if data.shape[1] == 2:
        return data[kind_tag].value_counts().idxmax()
    [split_feature, split_points] = get_the_best_split_feature(data, continuous_feature, kind_tag)
    if data[split_feature].nunique() == 1:
        return data[kind_tag].value_counts().idxmax()
    if split_feature not in continuous_feature:
        tree = {split_feature: {}}
        split_data = data.groupby(split_feature)
        for [i, item] in split_data:
            tree[split_feature][i] = GenerateTree(item.drop(split_feature, axis=1), kind_tag, continuous_feature)
    else:
        tree = {split_feature + "<=" + str(split_points[split_feature][0]) + "?": {}}
        tree[split_feature + "<=" + str(split_points[split_feature][0]) + "?"]["yes"] = GenerateTree(
            data[data[split_feature] <= split_points[split_feature][0]], kind_tag, continuous_feature)
        tree[split_feature + "<=" + str(split_points[split_feature][0]) + "?"]["no"] = GenerateTree(
            data[data[split_feature] > split_points[split_feature][0]], kind_tag, continuous_feature)
    return tree


def classify(sample, DecisionTree):
    """
    对样本进行分类
    :param DecisionTree: 决策树
    :param sample: 样本
    :return: 分类结果
    """
    if type(DecisionTree) != dict:
        return DecisionTree
    else:
        feature_name = list(DecisionTree.keys())[0]
        if "<=" in feature_name:
            feature_value = sample[feature_name.split("<=")[0]]
            if feature_value <= float(feature_name.split("<=")[1][:-1]):
                return classify(DecisionTree[feature_name]["yes"], sample)
            else:
                return classify(DecisionTree[feature_name]["no"], sample)
        else:
            feature_value = sample[feature_name]
            return classify(DecisionTree[feature_name][feature_value], sample)
