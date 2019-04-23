import numpy as np
import os
import csv
import glob

from arguments import *


def process_data_(x_input_dir, y_input_dir, activity_list):
    xx = np.empty([0, slide_win_size, carrier_nums], float)  # [0,window_size,90]

    # data import from csv
    input_csv_files = sorted(glob.glob(x_input_dir))
    for file in input_csv_files:
        print("input_file_name=", file)
        data = [[float(elm) for elm in v] for v in csv.reader(open(file, "r"))]
        tmp1 = np.array(data)
        x2 = np.empty([0, slide_win_size, carrier_nums], float)  # [0,window_size,90]

        # data import by slide window
        k = 0
        while k <= (len(tmp1) + 1 - 2 * slide_win_size):
            x = np.dstack(np.array(tmp1[k:k + slide_win_size, 0:carrier_nums]).T)
            x2 = np.concatenate((x2, x), axis=0)
            k += slide_steps
        xx = np.concatenate((xx, x2), axis=0)
    print(xx.shape)
    xx = xx.reshape(len(xx), -1)  # [++,window_size,90]->[++,window_size*90]

    # data import from csv
    annotation_csv_files = sorted(glob.glob(y_input_dir))
    column_nums = len(activity_list) + 1

    yy = np.empty([0, column_nums], float)
    for ff in annotation_csv_files:
        print("annotation_file_name=", ff)
        ano_data = [[str(elm) for elm in v] for v in csv.reader(open(ff, "r"))]
        tmp2 = np.array(ano_data)

        # data import by slide window
        y = np.zeros((int((len(tmp2) + 1 - 2 * slide_win_size) / slide_steps) + 1, column_nums))
        k = 0
        while k <= (len(tmp2) + 1 - 2 * slide_win_size):
            y_pre = np.stack(np.array(tmp2[k:k + slide_win_size]))

            label_dict = {}
            for l in activity_list:
                label_dict[l] = 0
            for j in range(slide_win_size):
                for mark in activity_list:
                    if y_pre[j] == mark:
                        label_dict[mark] += 1
            flag = False
            for ii, mark in enumerate(activity_list):
                if label_dict[mark] > slide_win_size * threshold / 100:
                    flag = True
                    t_ = np.zeros(shape=(column_nums,))
                    t_[ii + 1] = 1
                    y[int(k / slide_steps), :] = t_
            if not flag:
                t_ = np.zeros(shape=(column_nums,))
                t_[0] = 2
                y[int(k / slide_steps), :] = t_
            k += slide_steps

        yy = np.concatenate((yy, y), axis=0)  # [++,4]
    print(xx.shape, yy.shape)
    return (xx, yy)  # [++,window_size*90],[++,4]


for data_name in ALL_DATA_NAMES:

    # +++++++++++++++++++从标记文件的文件名中提取标签+++++++++++++++++++++
    activity_list = set()
    current_data_name = INPUT_RAW_DATA_PKG + data_name + '/'
    label_files = current_data_name + "anno_*.csv"
    label_files = sorted(glob.glob(label_files))
    for s in label_files:
        label_ = s.split('\\')[-1].split('_')[1]  # e.g. anno_run_*_*.csv  --> run
        activity_list.add(label_)
    activity_list = sorted(list(activity_list))
    print(data_name + '的标签: ', activity_list)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for label in activity_list:
        x_input_dir = current_data_name + "{0}*.csv".format(label)
        y_input_dir = current_data_name + "anno*{0}*.csv".format(label)
        data_file_path = INPUT_PROCESSED_DATA_PKG + data_name + '_input_files_win{0}_steps{1}/' \
            .format(slide_win_size, slide_steps)
        if not os.path.exists(data_file_path):
            os.makedirs(data_file_path)
        output_file_name_1 = "./{0}xx_{1}_{2}_{3}.csv".format(data_file_path, slide_win_size, threshold, label)
        output_file_name_2 = "./{0}yy_{1}_{2}_{3}.csv".format(data_file_path, slide_win_size, threshold, label)

        x, y = process_data_(x_input_dir, y_input_dir, activity_list)
        with open(output_file_name_1, "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(x)
        with open(output_file_name_2, "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(y)
    print(data_name + " finished!")
