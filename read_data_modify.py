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
    # print("csv file importing...")

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

        xx = xx.reshape((len(xx), slide_win_size, int(carrier_nums / 3), 3)).transpose((0, 2, 1, 3))
        # 1000 Hz to 500 Hz (To avoid memory error)
        # if is_data_halve:
        #     xx = xx[:, ::2, :carrier_nums]
        # xx = xx.reshape((len(xx), -1))

        x_dic[str(i)] = xx
        y_dic[str(i)] = yy

        # print("finished_" + str(i))

    return x_dic, y_dic


def read_data(train_data_dir, activity_list, is_save=False, data_name=None):
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

    if is_save is True:
        num_examples = len(x_train)
        first_perm = np.arange(num_examples)
        np.random.shuffle(first_perm)
        x_train = x_train[first_perm]
        y_train = y_train[first_perm]
        train_shape = x_train.shape
        test_shape = x_test.shape
        x_train.reshape((train_shape[0], train_shape[1]*train_shape[2]*train_shape[3]))
        x_test.reshape((test_shape[0], test_shape[1]*test_shape[2]*test_shape[3]))
        np.savez('train_test_dataset/' + data_name+'_train.npz', arr=x_train)
        np.savez('train_test_dataset/' + data_name+'_test.npz', arr=x_test)
        np.savez('train_test_dataset/' + data_name+'_train_labels.npz', arr=y_train)
        np.savez('train_test_dataset/' + data_name+'_test_labels.npz', arr=y_test)
    else:
        return x_train, y_train, x_test, y_test

# def visible_data(data_set, file_name):
#     path = INPUT_RAW_DATA_PKG + data_set + '/' + file_name
#     print('可视化数据文件路径：', path)
#
#     data = np.array(pd.read_csv(path, header=None))
#     time_steps, subcarriers = data.shape
#     subcarrier = int(subcarriers / 3)
#     data = data.reshape((time_steps, subcarrier, 3)).transpose((1, 0, 2))
#
#     # sum_data = np.sum(data, axis=2)[:, :100]
#
#     anta_1_data = data[:, :, 0]
#     # print(anta_1_data)
#     anta_2_data = data[:, :, 1]
#     anta_3_data = data[:, :, 2]
#
#     plt.ylim(0, subcarrier)
#     plt.xlim([0, 3000])
#
#     plt.imshow(anta_1_data)
#     plt.show()
#     # plt.savefig('test.png')
#
#
# visible_data('vData2', 'run_chen_1.csv')

# for data_name in ALL_DATA_NAMES:
#     print(data_name + ' =' * 20)
#     x_train, y_train, x_test, y_test = read_data(data_name)
#     print('x_train len: ', len(x_train))
#     print('y_train len: ', len(y_train))
#     print('=' * 30)
