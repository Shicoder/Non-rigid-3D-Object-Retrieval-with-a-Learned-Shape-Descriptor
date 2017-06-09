

import numpy as np
from sklearn.preprocessing import normalize

def max_min_normalization(data_value):
    """ Data normalization using max value and min value

    Args:
        data_value: The data to be normalized
        data_col_max_values: The maximum value of data's columns
        data_col_min_values: The minimum value of data's columns
    """
    data_shape = data_value.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]

    for i in xrange(0, data_rows, 1):
        for j in xrange(0, data_cols, 1):
            data_value[i][j] = \
                (data_value[i][j] - np.min(data_value[i,:])) / \
                (np.max(data_value[i,:]) - np.min(data_value[i,:]))
    return data_value

data=np.array([[1,1,1],[4,5,6]])
data = normalize(data,axis=1,copy=False)
# data=max_min_normalization(data)
print data