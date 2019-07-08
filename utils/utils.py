import itertools
from builtins import print
import glob
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib
from arguments import *

matplotlib.use('agg')
import matplotlib.pyplot as plt
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = 'Arial'

import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return directory_path
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if y_true_val is not None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def compute_metrics_(y_true, y_pred):
    res = pd.DataFrame(data=np.zeros((1, 3), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    if lr is True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def compute_metrics(output_directory, y_pred, y_true):
    df_metrics = compute_metrics_(y_true, y_pred)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    return df_metrics


def choose_best_model(ckp_path):
    # 找到data_name数据集的最好训练模型----------------------------------------
    ckp_metric_file_path = ckp_path + '/*.h5f'
    paths = sorted(glob.glob(ckp_metric_file_path))

    best_val_model_path = None
    temp_min_val_loss = np.inf
    temp_max_val_acc = -np.inf
    #
    for path in paths:
        temp_split = path.split('-')
        min_val_loss = float(temp_split[3][4:])
        max_val_acc = float(temp_split[-1].split('.')[0][7:])
        if min_val_loss < temp_min_val_loss and max_val_acc >= temp_max_val_acc:
            temp_min_val_loss = min_val_loss
            temp_max_val_acc = max_val_acc
            best_val_model_path = path
    print('best model: ', best_val_model_path)
    return best_val_model_path


def save_confusion(true_label, pred_label, classes, save_path='/'):
    lmr_matrix = confusion_matrix(true_label, pred_label)
    acc_score = accuracy_score(true_label, pred_label)

    plt.imshow(lmr_matrix, interpolation='nearest', cmap=plt.cm.Blues,aspect="auto")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,fontsize=5)
    plt.yticks(tick_marks, classes,fontsize=5)
    plt.xlabel('Pre label')
    plt.ylabel('True label')
    # lmr_matrix = lmr_matrix.astype('float') / lmr_matrix.sum(axis=1)[:, np.newaxis]
    # fmt = '.2f'
    # thresh = lmr_matrix.max() / 2.
    # for i, j in itertools.product(range(lmr_matrix.shape[0]), range(lmr_matrix.shape[1])):
    #     plt.text(j, i, format(lmr_matrix[i, j], fmt),
    #              horizontalalignment="center",
    #              color="black" if lmr_matrix[i, j] > thresh else "red")
    #
    # for i, j in itertools.product(range(lmr_matrix.shape[0]), range(lmr_matrix.shape[1])):
    #     plt.text(j, i, lmr_matrix[i, j], fontsize=4)
    plt.title('confusion matrix acc={:.3f}'.format(acc_score), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path + 'confusion.png')


def generate_arrays_from_file(data_name, batch_size):
    while 1:
        f_train = open('train_test_dataset/' + data_name + '_train.csv')
        f_label = open('train_test_dataset/' + data_name + '_train_labels.csv')

        cnt = 0
        X = []
        Y = []
        for (train, label) in zip(f_train, f_label):
            # create Numpy arrays of input data
            # and labels, from each line in the file
            train = [float(v) for v in str(train).split(',')]
            X.append(train)
            label = [int(v) for v in str(label).split(' ')]
            Y.append(label)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(X).reshape(len(X), int(carrier_nums / 3), slide_win_size, 3), np.array(Y))
                X = []
                Y = []
    f.close()
