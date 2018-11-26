import numpy as np


def reject_outliers(data):
    """
    Function uses 3-sigma rule with probability 0.9973 of normally distributed expected value
    Takes numerical values and finds anomalies (outliers) in case if it's presented
    If defined value is beyond 3-sigma interval this is outlier.
    :param: numerical data
    :return: interval, outliers, filtered row without outliers if they are or initial row
    """
    mean = np.mean(data)
    std = np.std(data)
    data_filtered = data[(data > mean-3*std) & (data < mean+3*std)]
    data_anomaly = data[(data < mean-3*std) | (data > mean+3*std)]
    interval = list(map(lambda x: round(x, 5), [mean-3*std, mean+3*std]))
    if data_anomaly.size == 0:
        print("Anomalies were not detected")
    else:
        print("Anomalies were detected")
    return interval, data_anomaly, data_filtered


if __name__ == '__main__':
    # test data
    ts = np.array([0.0489, 0.0493, 0.0531, 0.0640, 0.0758, 0.0875, 0.0861, 0.0760, 0.0703, 0.0740, 0.0706, 0.2])
    inter, anomaly_list, anomaly_cleared = reject_outliers(ts)
    print('Interval is {}'.format(inter))
    print('Outliers {}'.format(anomaly_list))
    print('Cleared array {}'.format(anomaly_cleared))

