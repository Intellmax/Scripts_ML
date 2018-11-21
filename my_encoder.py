import numpy as np
import pandas as pd


def encoder(series, target=None):
    """Encoder function to label high-variable values (more than 30)
    Have 2 option:
    1 option - takes only one argument (series) to make result by frequency of values in series:
        - counts frequencies of all values
        - sorts frequencies descending
        - divides list of frequencies into quartiles
        - appoints the label to each referring quartile
    2 option - takes two arguments (series, target) to make result by sum of target column:
        - concatenates 2 series into one Dataframe
        - groups data by sum in target column
        - divides list of frequencies into quartiles
        - appoints the label to each referring quartile
    :type target: pandas.series
    :param: series (list of categorical values), target (None or list of categorical values)
    :return: numerical labeled series, key (key for labeled data)
    """

    d = {0.1: 1,
         0.2: 2,
         0.3: 3,
         0.4: 4,
         0.5: 5,
         0.6: 6,
         0.7: 7,
         0.8: 8,
         0.9: 9,
         1: 10}
    if target is None:
        frequency = series.value_counts(sort=True, ascending=False)
        quart = dict(frequency.quantile(list(d.keys())))
        temp = []
        for i in frequency:
            for j in quart:
                if i <= quart[j]:
                    temp.append(j)
                    break
        l1 = np.array([d[j] for i in temp for j in d if i == j])
        key = dict(list(zip(frequency.index, l1)))
        result = np.array([key.get(i) for i in series])
        return result, key
    else:
        table_frame = pd.concat([series, target], axis=1)
        target_group = table_frame.groupby(table_frame.columns[0]).sum()
        target_frequency = target_group.iloc[:, 0]
        target_frequency.index.name = None
        quart = dict(target_frequency.sort_values(ascending=False).quantile(list(d.keys())))
        temp = []
        for i in target_frequency:
            for j in quart:
                if i <= quart[j]:
                    temp.append(j)
                    break
        l1 = np.array([d[j] for i in temp for j in d if i == j])
        key = dict(list(zip(target_frequency.index, l1)))
        result = np.array([key.get(i) for i in series])
        return result, key


def decoder(series, key):
    """Decoder function which unlabels high-variable values (more than 30)
    Returns original series (grouped) from labelled ones
    Transforms vice-versa to categorical types
    :param: labeled numerical series, key (key for labeled data)
    :return: transformed categorical grouped series
    """
    key_vv = {v: [i for i in key.keys() if key[i] == v] for k, v in key.items()}
    key_res = [key_vv[i] for i in series if i in key_vv]
    result = [i if len(i) > 1 else i[0] for i in key_res]
    return result

# example


if __name__ == '__main__':
    series = pd.Series(data=['usa', 'usa', 'usa', 'jpn', 'jpn', 'jpn', 'jpn', 'jpn', 'fra', 'fra', 'sgp', 'sgp'])
    print('Original series', series)
    series_labeled, key_table = encoder(series)
    print('Series encoded', series_labeled)
    print('Key with labels', key_table)
    print('Decoded series', decoder(series_labeled, key_table))
