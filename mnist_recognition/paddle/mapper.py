import numpy as np


##根据分类概率给出预测数字0-9
def map_number(pre_array):
    pred = np.argsort(pre_array)[-1]
    return pred