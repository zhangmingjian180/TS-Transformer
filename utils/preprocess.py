
def split_matrix(matrix, split_length, win_length):
    """
    input:
        matrix:
            type: ndarray
            shape: e.g (62, 40401)
        split_length:
            the lenght of every split.
            type: int
        win_length:
            type: int
    output:
        m_list:
            type: list
            shape: len, e.g 101
            elem_type: ndarray
            shape: e.g (62, 100)
    """
    matrix_length = matrix.shape[1]
    
    m_list = []
    for i in range(split_length, matrix_length, win_length):
        m_list.append(matrix[:, i-split_length:i])
    
    return m_list


def find_same_str(str1, str2):
    """
    input:
        str1: e.g abc_eeg1
        str2: e.g abc_eeg2
    output:
        same_string: e.g abc_eeg
    note:
        output value not end with digit.
    """
    s = []
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            s.append(str1[i])

    while s[-1].isdigit():
        s.pop()
    same_string = "".join(s)
    return same_string


def get_sorted_keys(keys):
    """
    input:
        keys:
            the keys of mat dict.
            type: list
    output:
        varname_list:
            type: list
    """
    # create keys list
    varname_list = []
    for varname in keys:
        if varname[-1].isdigit():
            varname_list.append(varname)

    # sort list
    if len(varname_list) > 1:
        same_string = find_same_str(varname_list[0], varname_list[1])
        for i in range(len(varname_list)):
            if varname_list[i] != same_string + str(i+1):
                index = varname_list.index(same_string + str(i+1))
                varname_list[index] = varname_list[i]
                varname_list[i] = same_string + str(i+1)

    return varname_list
