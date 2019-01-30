import numpy as np


def softmax_agr(x):
    """
    Function aggressively (higher score gets higher weight) distributes weights of some scores based on exponential weighing
    :param x: -> list of scores
    :return: aggressively weighed scores
    """
    stdev = np.std(x)
    if stdev == 0:
        e_x = np.exp((x - np.max(x)))
        res = e_x / e_x.sum()
        return res
    else:
        e_x = np.exp((x - np.max(x))/np.std(x))
        res = e_x / e_x.sum()
        return res


if __name__ == '__main__':
    print(softmax_agr([1, 2, 3]))
    print(softmax_agr([1, 1.01, 1.05]))
    print(softmax_agr([0.003, 0.0035, 0.0039]))
