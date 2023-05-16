import numpy as np


def get_class_num(class_num, predictions):
    class_list = [0] * class_num
    for e in predictions:
        class_list[int(e)] += 1
    return class_list


def insert_zero(pos, conf_matrix):
    m = np.zeros((conf_matrix.shape[0] + 1, conf_matrix.shape[1] + 1), conf_matrix.dtype)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i < pos:
                if j < pos:
                    m[i][j] = conf_matrix[i][j]
                elif j > pos:
                    m[i][j] = conf_matrix[i][j-1]
                else:
                    pass
            elif i > pos:
                if j < pos:
                    m[i][j] = conf_matrix[i-1][j]
                elif j > pos:
                    m[i][j] = conf_matrix[i-1][j-1]
                else:
                    pass
    return m
