import sys
import numpy as np
np.log()

def test():
    '''
    numpy函数np.c_和np.r_学习使用
    '''
    data_list1 = [4, 6, 12, 6, 0, 3, 7]
    data_list2 = [1, 5, 2, 65, 6, 7, 3]
    data_list3 = [1, 5, 2, 65, 6]
    temp123 = np.r_[data_list1, data_list2, data_list3]
    print('temp123', temp123)

    temp = np.r_[data_list1, data_list2]
    temp = np.r_[temp, data_list3]
    print('temp   ', temp)
    x = []
    for i in [data_list1, data_list2, data_list3]:
        x = np.r_[x, i]
    print('x: ', x.astype(np.int))


if __name__ == '__main__':
    test()
