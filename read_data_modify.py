import glob
import numpy as np
import pandas as pd
from arguments import *


def return_labels(data_name):
    train_data_dir = INPUT_PROCESSED_DATA_PKG + data_name + \
                     '_input_files_win{0}_steps{1}/'.format(slide_win_size, slide_steps)
    label_patten = train_data_dir + 'xx_*.csv'
    labels_path = sorted(glob.glob(label_patten))
    activity_list = set()
    for s in labels_path:
        label_ = s.split('_')[-1].split('.')[0]
        activity_list.add(label_)
    activity_list = sorted(list(activity_list))
    return train_data_dir, activity_list


def csv_import(activity_list, train_data_path):
    x_dic = {}
    y_dic = {}
    print("csv file importing...")

    for i in activity_list:
        skip_row = 2  # Skip every 2 rows -> overlap 800ms to 600ms  (To avoid memory error)
        num_lines = sum(1 for l in open(train_data_path.format('xx', str(i))))
        skip_idx = [x for x in range(1, num_lines) if x % skip_row != 0]
        xx = np.array(
            pd.read_csv(train_data_path.format('xx', str(i)), header=None, skiprows=skip_idx))
        yy = np.array(
            pd.read_csv(train_data_path.format('yy', str(i)), header=None, skiprows=skip_idx))
        # xx = np.array(
        #     pd.read_csv(train_data_path.format('xx', str(i)), header=None))
        # yy = np.array(
        #     pd.read_csv(train_data_path.format('yy', str(i)), header=None))

        # eliminate the NoActivity Data
        rows, cols = np.where(yy > 0)
        xx = np.delete(xx, rows[np.where(cols == 0)], 0)
        yy = np.delete(yy, rows[np.where(cols == 0)], 0)

        xx = xx.reshape((len(xx), slide_win_size, int(carrier_nums / 3), 3)).transpose((0, 2, 1, 3)) #在这里控制数据输入形式
        # 1000 Hz to 500 Hz (To avoid memory error)
        # if is_data_halve:
        #     xx = xx[:, ::2, :carrier_nums]
        # xx = xx.reshape((len(xx), -1))

        x_dic[str(i)] = xx
        y_dic[str(i)] = yy

        print("finished_" + str(i))

    return x_dic, y_dic


def read_data(train_data_dir, activity_list):
    train_data_path = train_data_dir + '{0}_' + str(slide_win_size) + '_' + str(threshold) + '_{1}.csv'
    x_dict, y_dict = csv_import(activity_list, train_data_path)

    x_ = x_dict[activity_list[0]]
    x_ = np.roll(x_, int(len(x_) / kk), axis=0)
    y_ = y_dict[activity_list[0]]
    y_ = np.roll(y_, int(len(y_) / kk), axis=0)

    x_train = x_[int(len(x_) / kk):]
    y_train = y_[int(len(y_) / kk):]
    x_test = x_[:int(len(x_) / kk)]
    y_test = y_[:int(len(y_) / kk)]

    for mark in activity_list[1:]:
        x_ = x_dict[mark]
        x_ = np.roll(x_, int(len(x_) / kk), axis=0)
        y_ = y_dict[mark]
        y_ = np.roll(y_, int(len(y_) / kk), axis=0)

        x_1 = x_[int(len(x_) / kk):]
        x_train = np.r_[x_train, x_1]

        y_1 = y_[int(len(y_) / kk):]
        y_train = np.r_[y_train, y_1]

        x_2 = x_[:int(len(x_) / kk)]
        x_test = np.r_[x_test, x_2]

        y_2 = y_[:int(len(y_) / kk)]
        y_test = np.r_[y_test, y_2]

    y_train = y_train[:, 1:]
    y_test = y_test[:, 1:]

    return x_train, y_train, x_test, y_test


def _get_train_test_data(activity_list, train_data_path):

    print("Getting train and test data...")

    for idx, i in enumerate(activity_list):
        skip_row = 2  # Skip every 2 rows -> overlap 800ms to 600ms  (To avoid memory error)
        num_lines = sum(1 for l in open(train_data_path.format('xx', str(i))))
        skip_idx = [x for x in range(1, num_lines) if x % skip_row != 0]
        xx = np.array(
            pd.read_csv(train_data_path.format('xx', str(i)), header=None, skiprows=skip_idx))
        yy = np.array(
            pd.read_csv(train_data_path.format('yy', str(i)), header=None, skiprows=skip_idx))

        # eliminate the NoActivity Data
        rows, cols = np.where(yy > 0)
        xx = np.delete(xx, rows[np.where(cols == 0)], 0)
        yy = np.delete(yy, rows[np.where(cols == 0)], 0)

        xx = xx.reshape((len(xx), slide_win_size, int(carrier_nums / 3), 3)).transpose((0, 2, 1, 3))

        xx = np.roll(xx, int(len(xx) / kk), axis=0)
        yy = np.roll(yy, int(len(yy) / kk), axis=0)

        if idx == 0:
            x_train = xx[int(len(xx) / kk):]
            y_train = yy[int(len(yy) / kk):]
            x_test = xx[:int(len(xx) / kk)]
            y_test = yy[:int(len(yy) / kk)]
        else:
            x_1 = xx[int(len(xx) / kk):]
            x_train = np.r_[x_train, x_1]

            y_1 = yy[int(len(yy) / kk):]
            y_train = np.r_[y_train, y_1]

            x_2 = xx[:int(len(xx) / kk)]
            x_test = np.r_[x_test, x_2]

            y_2 = yy[:int(len(yy) / kk)]
            y_test = np.r_[y_test, y_2]

        print("finished_" + str(i))
    y_train = y_train[:, 1:]
    y_test = y_test[:, 1:]

    return x_train, y_train, x_test, y_test


def get_train_test_data(train_data_dir, activity_list, data_name=None):
    train_data_path = train_data_dir + '{0}_' + str(slide_win_size) + '_' + str(threshold) + '_{1}.csv'
    x_train, y_train, x_test, y_test = _get_train_test_data(activity_list, train_data_path)

    num_examples = len(x_train)
    first_perm = np.arange(num_examples)
    np.random.shuffle(first_perm)
    x_train = x_train[first_perm]
    y_train = y_train[first_perm]
    train_shape = x_train.shape
    test_shape = x_test.shape
    x_train = x_train.reshape((train_shape[0], train_shape[1] * train_shape[2] * train_shape[3]))
    x_test = x_test.reshape((test_shape[0], test_shape[1] * test_shape[2] * test_shape[3]))
    np.savetxt('train_test_dataset/' + data_name + '_train.csv', x_train, fmt='%.6f', delimiter=',')
    np.savez('train_test_dataset/' + data_name + '_test.npz', arr=x_test)
    np.savetxt('train_test_dataset/' + data_name + '_train_labels.csv', y_train, fmt='%d')
    np.savez('train_test_dataset/' + data_name + '_test_labels.npz', arr=y_test)
